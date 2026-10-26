import datetime as dt
import io
import sqlite3
import xml.etree.ElementTree as ET
from urllib.parse import quote
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

st.set_page_config(page_title='Pocket AI | Non-OTC research', page_icon='📈', layout='wide')
st.markdown('''<style>.block-container{max-width:1250px;padding-top:1.5rem}div[data-testid="stMetric"]{background:#17223510;padding:12px;border-radius:12px}</style>''', unsafe_allow_html=True)
FOREX = {'EUR/USD':'EURUSD=X','GBP/USD':'GBPUSD=X','USD/JPY':'JPY=X','AUD/USD':'AUDUSD=X','USD/CAD':'CAD=X','USD/CHF':'CHF=X','NZD/USD':'NZDUSD=X','EUR/JPY':'EURJPY=X','EUR/GBP':'EURGBP=X','EUR/CHF':'EURCHF=X','EUR/AUD':'EURAUD=X','EUR/CAD':'EURCAD=X','EUR/NZD':'EURNZD=X','GBP/JPY':'GBPJPY=X','GBP/CHF':'GBPCHF=X','GBP/AUD':'GBPAUD=X','GBP/CAD':'GBPCAD=X','AUD/JPY':'AUDJPY=X','AUD/CAD':'AUDCAD=X','AUD/NZD':'AUDNZD=X','CAD/JPY':'CADJPY=X','CHF/JPY':'CHFJPY=X','NZD/JPY':'NZDJPY=X'}
CRYPTO = {'BTC/USD':'BTC-USD','ETH/USD':'ETH-USD','SOL/USD':'SOL-USD','XRP/USD':'XRP-USD','LTC/USD':'LTC-USD','ADA/USD':'ADA-USD','DOGE/USD':'DOGE-USD','LINK/USD':'LINK-USD','BNB/USD':'BNB-USD','AVAX/USD':'AVAX-USD','DOT/USD':'DOT-USD'}
OTHER = {'Gold futures':'GC=F','Silver futures':'SI=F','Oil WTI futures':'CL=F','S&P 500 index':'^GSPC','Nasdaq 100 index':'^NDX','Dow Jones index':'^DJI','Apple':'AAPL','Microsoft':'MSFT','NVIDIA':'NVDA','Tesla':'TSLA','Amazon':'AMZN','Alphabet':'GOOGL','Meta':'META'}
GROUPS={'Forex':FOREX,'Crypto':CRYPTO,'Indices, commodities & stocks':OTHER}
FEATURES=['r1','r3','r5','ema_gap','rsi','atr_pct','body','range','hour_sin','hour_cos']
DB='pocket_ai_history.sqlite3'  # Temporary local storage; may reset on cloud restarts

def database():
    db=sqlite3.connect(DB,timeout=15,check_same_thread=False)
    db.execute('''CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY,asset TEXT,ticker TEXT,interval TEXT,created TEXT,expiry TEXT,direction TEXT,score REAL,entry REAL,outcome TEXT DEFAULT 'PENDING',exit_price REAL,holdout REAL)''')
    db.commit();return db

@st.cache_data(ttl=90,show_spinner=False)
def prices(ticker, interval):
    period={'1m':'7d','5m':'30d','15m':'60d','1h':'730d'}[interval]
    x=yf.download(ticker,period=period,interval=interval,progress=False,auto_adjust=True,threads=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    if x.empty: raise ValueError('Постачальник не повернув котирування')
    x=x[['Open','High','Low','Close']].dropna()
    x.index=pd.to_datetime(x.index,utc=True)
    return x[~x.index.duplicated(keep='last')].sort_index()

def features(df):
    x=df.copy();c=x.Close;d=c.diff();up=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();down=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    x['r1']=c.pct_change();x['r3']=c.pct_change(3);x['r5']=c.pct_change(5)
    x['ema_gap']=c.ewm(span=9,adjust=False).mean()/c.ewm(span=21,adjust=False).mean()-1
    x['rsi']=100-100/(1+up/(down+1e-12)); prev=c.shift(1)
    tr=pd.concat([x.High-x.Low,(x.High-prev).abs(),(x.Low-prev).abs()],axis=1).max(axis=1)
    x['atr_pct']=tr.rolling(14).mean()/c;x['body']=(x.Close-x.Open)/c;x['range']=(x.High-x.Low)/c
    h=x.index.hour;x['hour_sin']=np.sin(2*np.pi*h/24);x['hour_cos']=np.cos(2*np.pi*h/24)
    return x

def prediction(df,horizon):
    x=features(df)
    x['target']=np.where(x.Close.shift(-horizon)>x.Close,1,0).astype(float)
    x.iloc[-horizon:,x.columns.get_loc('target')]=np.nan
    valid=x.dropna(subset=FEATURES+['target'])
    if len(valid)<250: raise ValueError('Недостатньо свічок для тестування (потрібно 250+)')
    cut=int(len(valid)*.8)
    train=valid.iloc[:cut];test=valid.iloc[cut:]
    if train.target.nunique()<2: raise ValueError('Немає обох класів для навчання')
    m=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=8,l2_regularization=5,random_state=7)
    m.fit(train[FEATURES],train.target.astype(int))
    probs=m.predict_proba(test[FEATURES])[:,1]
    acc=accuracy_score(test.target.astype(int),(probs>=.5).astype(int))
    brier=brier_score_loss(test.target.astype(int),probs)
    m.fit(valid[FEATURES],valid.target.astype(int))
    p=float(m.predict_proba(x.dropna(subset=FEATURES).iloc[[-1]][FEATURES])[0,1])
    return p,float(acc),float(brier),len(test)

@st.cache_data(ttl=600,show_spinner=False)
def headlines(asset):
    try:
        q=quote(asset+' financial market economic news')
        r=requests.get('https://news.google.com/rss/search?q='+q+'&hl=en-US&gl=US&ceid=US:en',timeout=7,headers={'User-Agent':'Mozilla/5.0'})
        r.raise_for_status();root=ET.fromstring(r.content)
        out=[]
        for item in root.findall('./channel/item')[:8]:
            title=item.findtext('title','');url=item.findtext('link','');pub=item.findtext('pubDate','')
            if title and url:out.append((title,url,pub))
        return out
    except Exception:return []

def settle(db,ticker,df):
    pending=db.execute("SELECT id,expiry,direction,entry FROM signals WHERE ticker=? AND outcome='PENDING'",(ticker,)).fetchall()
    for ident,expiry,direction,entry in pending:
        future=df[df.index>=pd.Timestamp(expiry)]
        if future.empty:continue
        exit_price=float(future.iloc[0].Close)
        outcome='TIE' if exit_price==entry else ('WIN' if (exit_price>entry)==(direction=='CALL') else 'LOSS')
        db.execute('UPDATE signals SET outcome=?,exit_price=? WHERE id=?',(outcome,exit_price,ident))
    db.commit()

st.title('📈 Pocket AI — аналіз без OTC')
st.caption('Працює в браузері iPhone • дослідницький аналіз • зовнішні котирування Yahoo Finance, НЕ Pocket Option')
st.warning('Дослідницька версія. Немає гарантованих 70–80% точності. Новинні заголовки відображаються для ручної оцінки, але НЕ входять у ML-прогноз. Список активів — розширюваний приклад, а не підтверджений повний каталог Pocket Option.')
with st.sidebar:
    st.header('⚙️ Налаштування')
    group=st.selectbox('Категорія',list(GROUPS))
    asset=st.selectbox('Актив',list(GROUPS[group]))
    ticker=GROUPS[group][asset]
    interval=st.selectbox('Таймфрейм свічки',['1m','5m','15m','1h'],index=1)
    steps=st.selectbox('Експірація (кількість свічок)',[1,2,3,5],index=0)
    threshold=st.slider('Мінімальна оцінка моделі, % (не калібрована)',50,95,75)
    st.caption('Наявність у цьому списку не означає, що актив зараз відкритий у Pocket Option. У вихідні Forex часто закритий.')
    st.divider()
    custom=st.text_input('Додатковий Yahoo ticker (не OTC)',placeholder='наприклад EURSEK=X')
    if custom.strip():asset=custom.strip().upper();ticker=asset
    st.divider()
    st.subheader('🛡️ Перекриття — лише калькулятор')
    base=st.number_input('Початкова сума, $',min_value=0.1,max_value=100000.0,value=1.0,step=0.5)
    multiplier=st.selectbox('Множник', [2,3],index=1)
    covers=st.selectbox('Кількість перекриттів',[0,1,2,3],index=3)
    payout=st.slider('Виплата за виграш, % (введи фактичну)',10,100,80)
    run=st.button('🔍 Почати аналіз',type='primary',use_container_width=True)

tab1,tab2,tab3,tab4=st.tabs(['📊 Аналіз активу','📰 Новини','🧾 Журнал і статистика','🛡️ Перекриття'])
with tab1:
    st.subheader(asset)
    if run:
        with st.spinner('Завантажуємо дані та тестуємо модель...'):
            try:
                raw=prices(ticker,interval)
                # Drop the last potentially forming candle to prevent future leakage.
                df=raw.iloc[:-1].copy()
                step_minutes={'1m':1,'5m':5,'15m':15,'1h':60}[interval]
                last=df.index[-1];now=pd.Timestamp.now(tz='UTC')
                age=(now-last).total_seconds()/60
                stale=age>max(step_minutes*3,10)
                p,acc,brier,n=prediction(df,steps)
                confidence=max(p,1-p);direction='CALL' if p>=.5 else 'PUT'
                # No signals when market data are stale, accuracy is weak, or model score below threshold.
                decision=direction if not stale and confidence*100>=threshold and acc>=.53 else 'SKIP'
                close=float(df.Close.iloc[-1]);support=float(df.Low.tail(50).min());resistance=float(df.High.tail(50).max())
                st.session_state['last']={'asset':asset,'ticker':ticker,'interval':interval,'steps':steps,'decision':decision,'confidence':confidence,'acc':acc,'brier':brier,'n':n,'last':last,'age':age,'stale':stale,'close':close,'support':support,'resistance':resistance,'df':df}
                db=database();settle(db,ticker,df)
                if decision!='SKIP':
                    expiry=last+pd.Timedelta(minutes=step_minutes*steps)
                    if not db.execute('SELECT id FROM signals WHERE ticker=? AND interval=? AND created=? AND direction=?',(ticker,interval,last.isoformat(),direction)).fetchone():
                        db.execute('INSERT INTO signals(asset,ticker,interval,created,expiry,direction,score,entry,holdout) VALUES (?,?,?,?,?,?,?,?,?)',(asset,ticker,interval,last.isoformat(),expiry.isoformat(),direction,confidence,close,acc));db.commit()
            except Exception as e:
                st.error('Аналіз недоступний: '+str(e));st.session_state.pop('last',None)
    a=st.session_state.get('last')
    if a:
        if a['ticker']!=ticker or a['interval']!=interval:st.info('Натисніть «Почати аналіз» для нового активу.')
        else:
            st.success('Результат: '+a['decision']) if a['decision']!='SKIP' else st.info('Результат: ПРОПУСТИТИ — недостатньо підстав для сигналу.')
            c1,c2,c3,c4=st.columns(4)
            c1.metric('Оцінка моделі',f"{a['confidence']:.1%}")
            c2.metric('Точність holdout',f"{a['acc']:.1%}")
            c3.metric('Тестових прикладів',a['n'])
            c4.metric('Вік свічки, хв',f"{a['age']:.0f}")
            if a['stale']:st.error('Котирування застарілі або ринок закритий. Сигнали заблоковано.')
            st.caption('Оцінка моделі НЕ є каліброваною ймовірністю виграшу. Holdout — один хронологічний тест, не підтвердження стабільної прибутковості.')
            st.line_chart(a['df'].Close.tail(150),height=320)
            d1,d2,d3=st.columns(3)
            d1.metric('Останнє закриття (зовнішнє)',f"{a['close']:.6g}")
            d2.metric('Підтримка (50 свічок)',f"{a['support']:.6g}")
            d3.metric('Опір (50 свічок)',f"{a['resistance']:.6g}")
            st.write('Остання завершена свічка UTC:',str(a['last']))
            st.write('Brier score (нижче краще):',round(a['brier'],3))
            st.caption('Рівні — прості локальні екстремуми, не гарантія відскоку. Потрібно перевірити ціни й час на Pocket Option перед будь-яким рішенням.')
    else:st.info('Виберіть інструмент ліворуч і натисніть «Почати аналіз».')
with tab2:
    st.subheader('Останні заголовки: '+asset)
    st.caption('Інформаційний потік. Заголовки не оцінюються моделлю, не замінюють економічний календар і можуть надходити із затримкою.')
    if st.button('Оновити новини'):
        headlines.clear()
    news=headlines(asset)
    if news:
        for title,url,pub in news:st.markdown(f'- [{title}]({url}) — {pub}')
    else:st.warning('Новини недоступні. Не вважайте це підтвердженням відсутності важливих подій.')
with tab3:
    db=database()
    st.caption('На безкоштовному хмарному сервері локальний журнал може зникнути після перезапуску. Регулярно завантажуйте CSV.')
    log=pd.read_sql_query('SELECT asset,interval,created,expiry,direction,score,entry,outcome,exit_price,holdout FROM signals ORDER BY id DESC LIMIT 1000',db)
    if len(log):
        st.dataframe(log,use_container_width=True,hide_index=True)
        log['hour']=pd.to_datetime(log.created,utc=True).dt.floor('h')
        stats=log.groupby('hour').agg(signals=('outcome','size'),wins=('outcome',lambda x:(x=='WIN').sum()),losses=('outcome',lambda x:(x=='LOSS').sum()),ties=('outcome',lambda x:(x=='TIE').sum()),pending=('outcome',lambda x:(x=='PENDING').sum())).reset_index()
        stats['accuracy_%']=np.where(stats.wins+stats.losses>0,100*stats.wins/(stats.wins+stats.losses),np.nan)
        st.subheader('Статистика за годинами UTC');st.dataframe(stats.sort_values('hour',ascending=False),use_container_width=True,hide_index=True)
        st.download_button('⬇️ Завантажити CSV',log.to_csv(index=False).encode('utf-8-sig'),'signals.csv','text/csv')
    else:st.info('Журнал порожній. Сигнали додаються після аналізу, якщо пройшли фільтри.')
    st.caption('Результати оновлюються під час наступного аналізу того самого ticker. Перевірка використовує першу доступну свічку після експірації із зовнішнього джерела; це НЕ фактичний результат угоди на Pocket Option.')

with tab4:
    st.subheader('Калькулятор перекриттів — не рекомендація входу')
    st.warning('Перекриття не збільшують імовірність правильного прогнозу. Вони збільшують ризик і не повинні запускатися автоматично після програшу. Перед кожною новою угодою потрібен незалежний аналіз.')
    amounts=[float(base)*multiplier**i for i in range(covers+1)]
    total=sum(amounts)
    st.metric('Максимальна втрата всієї серії',f'${total:,.2f}')
    rows=[]
    lost=0.0
    for i,stake in enumerate(amounts):
        net=stake*payout/100-lost
        rows.append({'Етап':'Початковий' if i==0 else f'Перекриття {i}', 'Сума, $':round(stake,2),'Втрачено до етапу, $':round(lost,2),'Результат серії при виграші, $':round(net,2)})
        lost+=stake
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.caption('Приклад припускає, що всі попередні угоди програли, а виграшна повертає ставку та прибуток за вказаним payout. Виплати, котирування та мінімальні ставки на платформі можуть відрізнятися. Результати не враховують комісії та інші умови.')
    st.info('Сигнали та журнал працюють окремо від калькулятора. Жодне перекриття не вважається сигналом CALL/PUT без нового незалежного прогнозу.')

# --- Automatic cross-market scanner (research-only) ---
from concurrent.futures import ThreadPoolExecutor, as_completed

def scan_market(ticker, interval, horizon, minimum_score):
    raw = prices(ticker, interval)
    df = raw.iloc[:-1].copy()  # only completed bars
    if len(df) < 260:
        return None
    mins = {'1m': 1, '5m': 5, '15m': 15, '1h': 60}[interval]
    age = (pd.Timestamp.now(tz='UTC') - df.index[-1]).total_seconds() / 60
    if age > max(mins * 3, 10):
        return None
    p, acc, brier, n = prediction(df, horizon)
    confidence = max(p, 1 - p)
    if confidence * 100 < minimum_score or acc < .53:
        return None
    return dict(direction='CALL' if p >= .5 else 'PUT', confidence=confidence,
                holdout_accuracy=acc, brier=brier, test_n=n,
                price=float(df.Close.iloc[-1]), support=float(df.Low.tail(50).min()),
                resistance=float(df.High.tail(50).max()), bar_time=df.index[-1].isoformat(),
                age_minutes=age, expiry_minutes=mins*horizon)

st.divider()
st.header('🌐 Автосканер усіх підтримуваних ринків — без OTC')
st.caption('Не потрібно обирати пару: сканер паралельно перевіряє весь перелік нижче. Це зовнішні інструменти, а не підтверджений повний список Pocket Option. OTC у списку немає.')
with st.expander('Налаштування автосканера', expanded=True):
    scan_interval=st.selectbox('Свічка для сканування', ['5m','15m','1h','1m'], key='scan_interval')
    scan_horizon=st.selectbox('Експірація у свічках', [1,2,3,5], key='scan_horizon')
    scan_threshold=st.slider('Мінімальний бал моделі (не калібрований), %', 50,95,75,key='scan_threshold')
    scan_categories=st.multiselect('Категорії', list(GROUPS), default=list(GROUPS),key='scan_categories')
    scan_limit=st.slider('Скільки активів перевіряти за запуск', 5, len(FOREX)+len(CRYPTO)+len(OTHER), len(FOREX)+len(CRYPTO)+len(OTHER), key='scan_limit')
    start_scan=st.button('🚀 Просканувати всі ринки', type='primary', use_container_width=True)
if start_scan:
    universe=[(group,name,ticker) for group,assets in GROUPS.items() if group in scan_categories for name,ticker in assets.items()][:scan_limit]
    found=[]; errors=[]
    if not universe:
        st.warning('Оберіть хоча б одну категорію для сканування.')
    progress=st.progress(0)
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs={pool.submit(scan_market,ticker,scan_interval,scan_horizon,scan_threshold):(group,name,ticker) for group,name,ticker in universe}
        for i,job in enumerate(as_completed(jobs),1):
            group,name,ticker=jobs[job]
            try:
                result=job.result()
                if result: found.append({'Актив':name,'Категорія':group,'Напрямок':result['direction'],
                    'Оцінка моделі %':round(result['confidence']*100,1),
                    'Holdout %':round(result['holdout_accuracy']*100,1),
                    'Експірація хв':result['expiry_minutes'],'Ціна':result['price'],
                    'Підтримка':result['support'],'Опір':result['resistance'],
                    'Остання свічка UTC':result['bar_time']})
            except Exception as exc: errors.append((name,str(exc)[:100]))
            progress.progress(i/max(len(universe),1))
    found.sort(key=lambda x:(x['Оцінка моделі %'],x['Holdout %']),reverse=True)
    st.session_state['scan_results']=found
    st.session_state['scan_meta']=(len(universe),len(errors),pd.Timestamp.now(tz='UTC').isoformat())
    if errors:
        with st.expander(f'Не вдалося перевірити {len(errors)} активів'):
            st.dataframe(pd.DataFrame(errors,columns=['Актив','Причина']),hide_index=True)
if 'scan_results' in st.session_state:
    found=st.session_state['scan_results']
    count,failed,stamp=st.session_state['scan_meta']
    st.caption(f'Останнє сканування: {stamp} · перевірено {count} · помилок {failed} · кандидатів {len(found)}')
    if found:
        st.dataframe(pd.DataFrame(found),use_container_width=True,hide_index=True)
        st.download_button('⬇️ Завантажити результати',pd.DataFrame(found).to_csv(index=False).encode('utf-8-sig'),'market_scan.csv','text/csv')
        st.warning('Це кандидати для дослідження, НЕ перевірені торгові сигнали. Новини не враховані у моделі, оцінка не калібрована, а ціни можуть не збігатися з Pocket Option. Не відкривайте угоди на підставі лише цього рейтингу.')
    else:st.info('Сильних кандидатів за вибраними фільтрами немає або дані недоступні. Програма не вигадує сигнали.')
else:
    st.info('Натисніть «Просканувати всі ринки», щоб автоматично порівняти активи. Для нового сканування натисніть кнопку ще раз.')
