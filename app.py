import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import logging
import plotly.express as px

# Настройка страницы
st.set_page_config(page_title="MVP мониторинга выручки", layout="wide")
st.title("📊 MVP мониторинга экспортной выручки")
st.markdown("---")

# Инициализация session_state для хранения данных
if 'df' not in st.session_state:
    st.session_state.df = None
if 'clean_log' not in st.session_state:
    st.session_state.clean_log = {}

# Функция для очистки данных
def clean_data(df):
    """
    Очищает загруженный DataFrame и возвращает лог изменений.
    Адаптировано под конкретный файл ved_export_sample.csv.
    """
    log = {}
    df_clean = df.copy()

    # Логируем исходное состояние
    log['initial_rows'] = len(df_clean)
    log['initial_columns'] = list(df_clean.columns)

    # 1. Приведение заголовков к нижнему регистру, латинице и замене пробелов/спецсимволов
    new_columns = {
        'Контрагент': 'contractor',
        'Судно': 'vessel',
        'Страна': 'country',
        'ИНКОТЕРМС': 'incoterms',
        'Банк': 'bank',
        'Канал продаж': 'sales_channel',
        'Номер инвойса': 'invoice_id',
        'Дата инвойса или его отправки': 'invoice_date',
        'Сумма инвойса': 'invoice_amount',
        'Валюта инвойса': 'invoice_currency',
        'Состояние инвойса': 'status',
        'Дата фактического зачисления': 'payment_date',
        'Сумма фактической оплаты': 'payment_amount',
        'кросс-курс': 'cross_rate',
        'Менеджер': 'manager',
        'Комментарий': 'comment'
    }
    df_clean.rename(columns=new_columns, inplace=True)
    log['columns_renamed'] = list(df_clean.columns)

    # 2. Удаление дубликатов по 'invoice_id'
    duplicates = df_clean.duplicated(subset=['invoice_id'], keep='first').sum()
    df_clean = df_clean.drop_duplicates(subset=['invoice_id'], keep='first')
    log['duplicates_removed'] = duplicates

    # 3. Очистка и преобразование дат
    date_columns = ['invoice_date', 'payment_date']
    for col in date_columns:
        if col in df_clean.columns:
            # Пробуем разные форматы дат, деньfirst важно указать
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', dayfirst=True)
            invalid_dates = df_clean[col].isna().sum()
            log[f'{col}_invalid_dates'] = invalid_dates

    # 4. Очистка и преобразование сумм (специфичная для данного файла)
    amount_columns = ['invoice_amount', 'payment_amount']
    for col in amount_columns:
        if col in df_clean.columns:
            # Удаляем пробелы, запятые, символы валют, приближения и лишние пробелы
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace('≈', '', regex=False)  # Удаляем символ приближения
                .str.replace('usd', '', case=False, regex=False)  # Удаляем текст 'usd'
                .str.replace(',', '.', regex=False)  # Заменяем запятые на точки
                .str.replace(' ', '', regex=False)   # Удаляем все пробелы
                .str.replace('₽', '', regex=False)   # Удаляем символ рубля
                .str.replace('$', '', regex=False)   # Удаляем символ доллара
            )
            # Преобразуем в число, ошибки станут NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            invalid_amounts = df_clean[col].isna().sum()
            log[f'{col}_invalid_values'] = invalid_amounts

    # 5. Заполнение пустого кросс-курса медианным значением
    if 'cross_rate' in df_clean.columns:
        median_rate = df_clean['cross_rate'].median()
        null_cross_rates = df_clean['cross_rate'].isna().sum()
        df_clean['cross_rate'] = df_clean['cross_rate'].fillna(median_rate)
        log['cross_rate_nulls_filled'] = null_cross_rates
        log['cross_rate_median_value'] = median_rate

    # 6. Приведение статусов и других строковых полей к единому регистру и удаление пробелов
    string_columns = ['status', 'contractor', 'country', 'manager']
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
            log[f'{col}_normalized'] = True

    log['final_rows'] = len(df_clean)
    return df_clean, log

# Функция для расчета эвристики риска - ИСПРАВЛЕННАЯ
def calculate_risk_score(row):
    """
    Рассчитывает оценку риска (0-100) на основе эвристики.
    """
    score = 0
    today = pd.Timestamp.now().normalize()

    # 1. Просрочка (считаем due_date = invoice_date + 30 дней)
    if pd.notna(row.get('invoice_date')):
        due_date = row['invoice_date'] + pd.Timedelta(days=30)
        days_overdue = (today - due_date).days
        if days_overdue > 0:
            score += 50  # Базовый штраф за просрочку
            score += min(days_overdue * 5, 50)  # Доп. штраф за каждый день

    # 2. Крупный инвойс (используем сумму в RUB)
    if 'amount_rub' in row and row['amount_rub'] > df_clean['amount_rub'].median():
        score += 10

    # 3. Статус
    status = row.get('status', '')
    if status not in ['оплачен', 'closed', 'paid']:
        score += 15

    # 4. Страна (пример: считаем 'египет' рискованной)
    if row.get('country') == 'египет':
        score += 15

    return min(score, 100)

# Функция для создания Excel-отчета
# Функция для создания Excel-отчета - ИСПРАВЛЕННАЯ
def create_excel_report(df, kpi_data):
    """
    Создает Excel-отчет с несколькими листами.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Лист с исходными данными (добавляем колонку amount_rub)
        df_to_export = df.copy()
        if 'amount_rub' not in df_to_export.columns:
            df_to_export['amount_rub'] = df_to_export.apply(
                lambda x: x['invoice_amount'] * x['cross_rate'] if x['invoice_currency'] != 'RUB' else x['invoice_amount'],
                axis=1
            )
        df_to_export.to_excel(writer, sheet_name='Данные', index=False)

        # Лист с рисками (только инвойсы с высоким риском)
        high_risk_df = df_to_export[df_to_export['risk_score'] >= 50]
        high_risk_df.to_excel(writer, sheet_name='Высокий риск', index=False)

        # Лист с KPI
        kpi_df = pd.DataFrame.from_dict(kpi_data, orient='index', columns=['Значение'])
        kpi_df.to_excel(writer, sheet_name='KPI')

        # Получаем workbook и worksheet для форматирования
        workbook = writer.book
        worksheet_risk = writer.sheets['Высокий риск']

        # Добавляем форматирование для листа с рисками
        red_format = workbook.add_format({'bg_color': '#FFC7CE'})
        worksheet_risk.conditional_format('A1:Z1000', {'type': 'formula',
                                                       'criteria': '=$F1>=70',
                                                       'format': red_format})

    processed_data = output.getvalue()
    return processed_data

# --- INTERFACE ---

# Блок загрузки файла
uploaded_file = st.file_uploader("Загрузите файл с данными (CSV или Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Загрузка данных
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)

        # Очистка данных
        with st.spinner('Очищаем данные...'):
            df_clean, clean_log = clean_data(df)
            st.session_state.df = df_clean
            st.session_state.clean_log = clean_log

        # Показываем лог очистки
        st.success("Данные успешно загружены и очищены!")
        with st.expander("Показать лог очистки данных"):
            st.json(clean_log)

        # Расчет медианы для суммы после очистки
        median_amount = df_clean['amount'].median() if 'amount' in df_clean.columns else 0

        # Конвертируем все суммы в рубли для всего DataFrame
        df_clean['amount_rub'] = df_clean.apply(
            lambda x: x['invoice_amount'] * x['cross_rate'] if x['invoice_currency'] != 'RUB' else x['invoice_amount'],
            axis=1
        )

        # Добавляем столбец с риском
        with st.spinner('Расчитываем риски...'):
            # Сначала применяем функцию без учета суммы
            df_clean['risk_score'] = df_clean.apply(calculate_risk_score, axis=1)
            # Теперь добавляем риск за крупную сумму
            if 'amount' in df_clean.columns:
                df_clean.loc[df_clean['amount'] > median_amount, 'risk_score'] += 10
                df_clean['risk_score'] = df_clean['risk_score'].clip(0, 100)

        # --- БЛОК ФИЛЬТРОВ ---
        st.sidebar.header("Фильтры")

        # Фильтр по риску
        min_risk = st.sidebar.slider("Минимальный риск", 0, 100, 50)
        df_filtered = df_clean[df_clean['risk_score'] >= min_risk]

        # Фильтр по стране
        if 'country' in df_filtered.columns:
            countries = ['Все'] + list(df_filtered['country'].unique())
            selected_country = st.sidebar.selectbox("Страна", countries)
            if selected_country != 'Все':
                df_filtered = df_filtered[df_filtered['country'] == selected_country]

        # Фильтр по статусу
        if 'status' in df_filtered.columns:
            statuses = ['Все'] + list(df_filtered['status'].unique())
            selected_status = st.sidebar.selectbox("Статус", statuses)
            if selected_status != 'Все':
                df_filtered = df_filtered[df_filtered['status'] == selected_status]

        # --- БЛОК KPI ---
        st.header("Ключевые показатели")
        col1, col2, col3, col4 = st.columns(4)

        # Инициализация переменных
        total_expected = 0
        total_overdue = 0
        total_amount = 0
        overdue_ratio = 0
        total_invoices = len(df_filtered)
        top_debtors = pd.DataFrame()

        # Создаем колонку с предполагаемой датой оплаты
        df_filtered['due_date'] = df_filtered['invoice_date'] + pd.Timedelta(days=30)

        # Текущая дата
        today = datetime.now()

        # Расчет KPI с учетом валют
        if 'invoice_amount' in df_filtered.columns and 'invoice_currency' in df_filtered.columns:

            # Конвертируем все суммы в рубли для корректного суммирования
            df_filtered['amount_rub'] = df_filtered.apply(
                lambda x: x['invoice_amount'] * x['cross_rate'] if x['invoice_currency'] != 'RUB' else x[
                    'invoice_amount'],
                axis=1
            )

            # ОЖИДАЕМАЯ ВЫРУЧКА: сумма всех НЕоплаченных инвойсов
            unpaid_mask = (~df_filtered['status'].isin(['оплачен', 'paid', 'closed']))
            total_expected = df_filtered.loc[unpaid_mask, 'amount_rub'].sum()

            # Сумма просроченных платежей (в RUB)
            overdue_mask = (df_filtered['due_date'] < today)
            total_overdue = df_filtered.loc[overdue_mask, 'amount_rub'].sum()

            # Общая сумма всех инвойсов (в RUB)
            total_amount = df_filtered['amount_rub'].sum()

            # Доля просроченных
            if total_amount > 0:
                overdue_ratio = (total_overdue / total_amount) * 100

            # Топ-5 контрагентов по просрочке (в RUB)
            if 'contractor' in df_filtered.columns:
                overdue_df = df_filtered[df_filtered['due_date'] < today]
                if not overdue_df.empty:
                    top_debtors = overdue_df.groupby('contractor')['amount_rub'].sum().nlargest(5).reset_index()
                    top_debtors.columns = ['Контрагент', 'Сумма просрочки (RUB)']

        # Отображение KPI
        with col1:
            st.metric("Всего инвойсов", total_invoices)
        with col2:
            st.metric("Ожидаемая выручка", f"{total_expected:,.0f} ₽")
            st.caption("Сумма неоплаченных инвойсов")
        with col3:
            st.metric("Сумма просрочки", f"{total_overdue:,.0f} ₽")
        with col4:
            st.metric("Доля просрочки", f"{overdue_ratio:.1f}%")

        # Дополнительная статистика
        with st.expander("Детальная статистика"):
            col11, col22, col33 = st.columns(3)

            with col11:
                paid_mask = (df_filtered['status'].isin(['оплачен', 'paid', 'closed']))
                paid_amount = df_filtered.loc[paid_mask, 'amount_rub'].sum()
                st.metric("Оплаченные инвойсы", f"{paid_amount:,.0f} ₽")

            with col22:
                unpaid_count = len(df_filtered[~df_filtered['status'].isin(['оплачен', 'paid', 'closed'])])
                st.metric("Неоплаченные инвойсы", unpaid_count)

            with col33:
                overdue_count = len(df_filtered[df_filtered['due_date'] < today])
                st.metric("Просроченные инвойсы", overdue_count)

        # ОТОБРАЖЕНИЕ ТОПА ДОЛЖНИКОВ
        if not top_debtors.empty:
            st.subheader("Топ-5 контрагентов по сумме просрочки (в рублях)")
            st.dataframe(top_debtors, use_container_width=True)

            # Визуализация
            fig = px.bar(top_debtors,
                         x='Контрагент',
                         y='Сумма просрочки (RUB)',
                         title='Топ-5 контрагентов по сумме просрочки',
                         color='Контрагент')
            st.plotly_chart(fig, use_container_width=True)

            # Дополнительная таблица с разбивкой по валютам (УПРОЩЕННАЯ ВЕРСИЯ)
            st.subheader("Детализация долгов по валютам (в рублях)")
            if 'contractor' in df_filtered.columns:
                currency_debt = df_filtered[df_filtered['due_date'] < today]
                if not currency_debt.empty:
                    # Группируем по контрагенту и суммируем в рублях
                    currency_breakdown = currency_debt.groupby('contractor')['amount_rub'].sum().reset_index()
                    currency_breakdown.columns = ['Контрагент', 'Сумма просрочки (RUB)']
                    currency_breakdown['Сумма просрочки (RUB)'] = currency_breakdown['Сумма просрочки (RUB)'].round(2)

                    st.dataframe(currency_breakdown, use_container_width=True)
                else:
                    st.info("Нет данных для детализации долгов")
            else:
                st.warning("Отсутствует колонка с контрагентами")

        else:
            st.info("Нет просроченных инвойсов для отображения топа должников.")

        # Дополнительная информация о валютах
        with st.expander("Информация о валютах и курсах"):
            st.write("**Использованные курсы конвертации:**")
            if 'invoice_currency' in df_filtered.columns and 'cross_rate' in df_filtered.columns:
                currency_rates = df_filtered.groupby('invoice_currency')['cross_rate'].mean().round(4)
                st.write(currency_rates)

            st.write("**Распределение инвойсов по валютам:**")
            if 'invoice_currency' in df_filtered.columns:
                currency_counts = df_filtered['invoice_currency'].value_counts()
                st.write(currency_counts)

            st.write("**Статусы инвойсов:**")
            if 'status' in df_filtered.columns:
                status_counts = df_filtered['status'].value_counts()
                st.write(status_counts)

        # ... (после блока с KPI в основном коде app.py)

        # --- ПРОВЕРКА ГИПОТЕЗ ---
        st.header("Проверка гипотез")

        # Создаем копию данных для анализа
        hypothesis_df = df_clean.copy()

        # Создаем колонку due_date для анализа
        hypothesis_df['due_date'] = hypothesis_df['invoice_date'] + pd.Timedelta(days=30)

        # Создаем признак просрочки
        hypothesis_df['is_overdue'] = (hypothesis_df['due_date'] < datetime.now()) & (
            ~hypothesis_df['status'].isin(['оплачен', 'paid', 'closed']))

        if 'country' in hypothesis_df.columns:

            # Анализ просрочек по странам - ИСПОЛЬЗУЕМ RUB
            country_overdue = hypothesis_df.groupby('country').agg(
                total_invoices=('invoice_id', 'count'),
                overdue_invoices=('is_overdue', 'sum'),
                avg_invoice_amount_rub=('amount_rub', 'mean')  # Используем рубли
            ).round(2)

            # Расчет доли просрочек
            country_overdue['overdue_ratio'] = (country_overdue['overdue_invoices'] / country_overdue[
                'total_invoices']) * 100
            country_overdue['overdue_ratio'] = country_overdue['overdue_ratio'].round(2)

            # Расчет средней суммы просрочки в RUB
            overdue_df = hypothesis_df[hypothesis_df['is_overdue'] == True]
            if not overdue_df.empty:
                country_overdue_amount = overdue_df.groupby('country')['amount_rub'].mean().round(2)
                country_overdue = country_overdue.join(country_overdue_amount.rename('avg_overdue_amount_rub'))
            else:
                country_overdue['avg_overdue_amount_rub'] = 0

            st.subheader("Анализ просрочек по странам (суммы в RUB)")
            st.dataframe(country_overdue.sort_values('overdue_ratio', ascending=False))

            # Визуализация
            col1, col2 = st.columns(2)

            with col1:
                st.write("Доля просроченных инвойсов по странам (%)")
                fig1 = px.bar(country_overdue.sort_values('overdue_ratio', ascending=False).reset_index(),
                              x='country', y='overdue_ratio',
                              title='Доля просрочек по странам')
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.write("Средняя сумма инвойса по странам (RUB)")
                fig2 = px.bar(country_overdue.sort_values('avg_invoice_amount_rub', ascending=False).reset_index(),
                              x='country', y='avg_invoice_amount_rub',
                              title='Средняя сумма инвойса по странам (RUB)')
                st.plotly_chart(fig2, use_container_width=True)

            # Проверка гипотезы о конкретной стране
            st.subheader("Проверка гипотезы")

            # Автоматически определяем страну с наибольшей долей просрочек
            if not country_overdue.empty:
                highest_risk_country = country_overdue['overdue_ratio'].idxmax()
                highest_risk_ratio = country_overdue['overdue_ratio'].max()
                median_ratio = country_overdue['overdue_ratio'].median()

                st.write(f"**Гипотеза:** Инвойсы из {highest_risk_country} имеют повышенный процент просрочки")

                if highest_risk_ratio > median_ratio * 1.5:  # На 50% выше медианного значения
                    st.success(
                        f"✅ Гипотеза **ПОДТВЕРЖДЕНА**: {highest_risk_country} имеет долю просрочек {highest_risk_ratio}%, что значительно выше медианного значения ({median_ratio}%)")
                elif highest_risk_ratio > median_ratio:
                    st.info(
                        f"⚠️ Гипотеза **ЧАСТИЧНО ПОДТВЕРЖДЕНА**: {highest_risk_country} имеет долю просрочек {highest_risk_ratio}%, что выше медианного значения ({median_ratio}%), но не значительно")
                else:
                    st.error(
                        f"❌ Гипотеза **НЕ ПОДТВЕРЖДЕНА**: {highest_risk_country} имеет долю просрочек {highest_risk_ratio}%, что не превышает медианное значение ({median_ratio}%)")

                # Дополнительная аналитика
                st.write("**Дополнительная аналитика:**")
                st.write(
                    f"- Страна с наименьшим риском: {country_overdue['overdue_ratio'].idxmin()} ({country_overdue['overdue_ratio'].min()}% просрочек)")
                st.write(f"- Общее количество стран: {len(country_overdue)}")
                st.write(f"- Средняя доля просрочек: {country_overdue['overdue_ratio'].mean():.2f}%")

        else:
            st.warning("Недостаточно данных для проверки гипотез. Отсутствуют необходимые колонки.")

        # Анализ по менеджерам - ИСПОЛЬЗУЕМ RUB
        if 'manager' in hypothesis_df.columns:
            st.subheader("Анализ просрочек по менеджерам (суммы в RUB)")

            manager_stats = hypothesis_df.groupby('manager').agg(
                total_invoices=('invoice_id', 'count'),
                overdue_invoices=('is_overdue', 'sum'),
                avg_amount_rub=('amount_rub', 'mean')  # Используем рубли
            ).round(2)

            manager_stats['overdue_ratio'] = (manager_stats['overdue_invoices'] / manager_stats['total_invoices']) * 100
            manager_stats = manager_stats.sort_values('overdue_ratio', ascending=False)

            st.dataframe(manager_stats)

            fig3 = px.bar(manager_stats.reset_index(),
                          x='manager', y='overdue_ratio',
                          title='Доля просрочек по менеджерам')
            st.plotly_chart(fig3, use_container_width=True)



        # --- БЛОК ТАБЛИЦ И ЭКСПОРТА ---
        st.header("Таблица инвойсов с рисками")
        st.dataframe(df_filtered, use_container_width=True)



        # Подготовка данных для отчета
        kpi_data = {
            'Ожидаемая выручка (30 дн.)': total_expected,
            'Сумма просрочки': total_overdue,
            'Доля просрочки (%)': overdue_ratio,
            'Общая сумма': total_amount,
            'Количество инвойсов': len(df_filtered)
        }

        # Кнопка экспорта
        excel_report = create_excel_report(df_filtered, kpi_data)
        st.download_button(
            label="📥 Скачать Excel-отчёт",
            data=excel_report,
            file_name="export_revenue_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Произошла ошибка при обработке файла: {e}")
else:
    st.info("👆 Загрузите CSV или Excel файл для начала анализа.")
