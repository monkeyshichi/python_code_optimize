ת���� ΢�Ź��ں� Python����֮��

# �����ȷʹ��Pandas��������Ŀ�������ٶȣ�

�������´����ݹ�������Python��Pandas��ʱ�ᷢ�ֺܶྪϲ��Pandas�����ݿ�ѧ�ͷ����������Խ��Խ��Ҫ�Ľ�ɫ�������Ƕ��ڴ�Excel��VBAת��Python���û���
 
���ԣ��������ݿ�ѧ�ң����ݷ���ʦ�����ݹ���ʦ��Pandas��ʲô�أ�Pandas�ĵ���Ķ����Ľ����ǣ�
 
�����١����������������ݽṹ���Դ��ô����ϵ�����ݺʹ��б�ǩ������ʱ����ֱ�ۡ���
 
���١����򵥺�ֱ�ۣ���Щ���Ǻܺõ����ԡ����㹹�����ӵ�����ģ��ʱ������Ҫ�ٻ������Ŀ���ʱ���ڵȴ����ݴ�����������ˡ��������Խ�����ľ�������ȥ������ݡ�

���ǣ�����˵Pandas����

��һ��ʹ��Pandasʱ����������˵��Pandas�Ǻܰ��Ľ������ݵĹ��ߣ�����Pandas̫���ˣ��޷�����ͳ�ƽ�ģ����һ��ʹ�õ�ʱ��ȷʵ��ˣ�������� 

���ǣ�Pandas�ǽ�����NumPy����ṹ֮�ϵġ��������ĺܶ����ͨ��NumPy����Pandas�Դ�����չģ���д����Щģ����Cython��д�����뵽C��������C��ִ�С���ˣ�Pandas��ҲӦ�úܿ���� 

��ʵ�ϣ�ʹ��������ȷ�Ļ���Pandasȷʵ�ܿ졣 

��ʹ��Pandasʱ��ʹ�ô���python��ʽ���벢������Ч�ʵ�ѡ�񡣺�NumPyһ����PandasרΪ��������������ƣ�������һ��ɨ������ɶ����л������ݼ��Ĳ���������������ÿ����Ԫ���ĳһ�����ֱ�������Ϊ��Ӧ����Ϊ����ѡ��

# ���̳�

��˵���£����̳̲���������ι����Ż�Pandas���롣��ΪPandas����ȷ��ʹ�����Ѿ��ܿ��ˡ����⣬�Ż�����ͱ�д�����Ĵ���֮��Ĳ����Ǿ޴�ġ�
 
����һƪ���ڡ���γ������Pandas���õ�ǿ�����������ֵ����ԡ���ָ�������⣬�㽫ѧϰ��һЩʵ�õĽ�ʡʱ��ļ��ɡ�����ƪ�̳��У��㽫ѧϰ���� 

�� ʹ��datetimeʱ���������ݵ�����

�� �������������Ч�ʵķ���

�� ����HDFStore��ʡʱ��
 
�ڱ����У��ĵ���ʱ���������ݽ���������ʾ�����⡣�������ݺ����ǽ����˽����Ч�ʵķ���ȡ�����ս��������Pandas�û����ԣ����ж��ַ���Ԥ�������ݡ������ⲻ��ζ�����з����������ڸ��󡢸����ӵ����ݼ���

��ע��
Github Դ�����ĩ��1��

�����ߡ�
Python 3��Pandas 0.23.1

# ����

����ʹ����Դ���ĵ�ʱ���������ݼ���һ����Դ���ܳɱ������ڲ�ͬʱ��εĵ�۲�ͬ�������Ҫ����ʱ�εĺĵ������϶�Ӧʱ�εĵ�ۡ� 

��CSV�ļ��п��Զ�ȡ���������ݣ�����ʱ��͵������ģ�ǧ�ߣ�

![testpic](https://github.com/monkeyshichi/python_code_optimize/blob/master/python_speedup_codesample/imgfolder/tb1.png)

ÿ�������ж�����ÿСʱ�ĵ������ݣ������������8760��356��24�������ݡ�ÿ�е�Сʱ���ݱ�ʾ����Ŀ�ʼʱ�䣬���1/1/13 0��00������ָ1��1�ŵ�1��Сʱ�ĺĵ������ݡ� 
# ��Datetime���ʡʱ�� 
������Pandas��һ��I/O������ȡCSV�ļ���
```
>>> import pandas as pd
>>> pd.__version__
'0.23.1'

>>> df = pd.read_csv('�ļ�·��')
>>> df.head()
     date_time  energy_kwh
0  1/1/13 0:00       0.586
1  1/1/13 1:00       0.580
2  1/1/13 2:00       0.572
3  1/1/13 3:00       0.596
4  1/1/13 4:00       0.592
```

��������ȥͦ�ã������и�С���⡣Pandas ��NumPy�и���������dtypes������粻ָ�������Ļ���date_time���н��ᱻ��ΪĬ����object��
```
>>> df.dtypes
date_time      object
energy_kwh    float64
dtype: object

>>> type(df.iat[0, 0])
str
```
Ĭ����object������str������������Ҳ���������������ĳһ���������͡��ַ���str���͵����������ݴ������Ƿǳ���Ч�ģ�ͬʱ�ڴ�Ч��Ҳ�ǵ��µġ� 

Ϊ�˴���ʱ���������ݣ���Ҫ��date_time�и�ʽ��Ϊdatetime������飬Pandas ��������������Ϊʱ���Timestamp����Pandas���и�ʽ���൱�򵥣�
```
>>> df['date_time'] = pd.to_datetime(df['date_time'])
>>> df['date_time'].dtype
datetime64[ns]
```
���ˣ��µ�df��CSV file���ݻ���һ�����������к�һ��������
```
>>> df.head()
               date_time    energy_kwh
0    2013-01-01 00:00:00         0.586
1    2013-01-01 01:00:00         0.580
2    2013-01-01 02:00:00         0.572
3    2013-01-01 03:00:00         0.596
4    2013-01-01 04:00:00         0.592
```

������������׶���������ִ���ٶ�����أ���������ʹ����timingװ���������ｫװ������Ϊ@timeit�����װ����ģ����Python��׼���е�timeit.repeat() ���������������Է��غ����Ľ�������Ҵ�ӡ����ظ����Ե�ƽ������ʱ�䡣Python��timeit.repeat() ֻ���ص���ʱ�������������غ�������� 

��װ����@timeit���ں����Ϸ���ÿ�����к���ʱ����ͬʱ��ӡ�ú���������ʱ�䡣
```
>>> @timeit(repeat=3, number=10)
... def convert(df, column_name):
...     return pd.to_datetime(df[column_name])

>>> # Read in again so that we have `object` dtype to start 
>>> df['date_time'] = convert(df, 'date_time')
Best of 3 trials with 10 function calls per trial:
Function `convert` ran in average of 1.610 seconds.
```

�������Σ�����8760�����ݺ�ʱ1.6�롣���ƺ�ûɶë�������ǵ������������ݼ�ʱ������������Ƶ�ĵ�����ݣ�����ÿ���ӵĵ������ȥ����һ������ܳɱ���������������ڶ�60��������ζ������Ҫ��Լ90��ȥ�ȴ�����Ľ��������е��̲����ˡ�
 
ʵ���ϣ����߹�������Ҫ����330��վ���ȥ10���ÿСʱ�������ݡ����ϱߵķ�������Ҫ88�������ʱ���еĸ�ʽ��ת����
 
�и���ķ�����һ����˵��Pandas���Ը����ת��������ݡ��ڱ����У�ʹ�ø�ʽ������csv�ļ����ض���ʱ���ʽ����Pandas��to_datetime�У����Դ������������Ч�ʡ�

```
>>> @timeit(repeat=3, number=100)
>>> def convert_with_format(df, column_name):
...     return pd.to_datetime(df[column_name],
...                           format='%d/%m/%y %H:%M')
Best of 3 trials with 100 function calls per trial:
Function `convert_with_format` ran in average of 0.032 seconds.
```
�µĽ����Σ�0.032�룬�ٶ�������50��������֮ǰ330վ������ݴ���ʱ���ʡ��86���ӡ�
 
һ����Ҫע���ϸ����CSV�е�ʱ���ʽ����ISO 8601��ʽ��YYYY-mm-dd HH��MM�����û��ָ����ʽ��Pandas��ʹ��dateuil����ÿ���ַ�����ʽ�����ڸ�ʽ�����෴�����ԭʼ��ʱ���ʽ�Ѿ���ISO 8601��ʽ�ˣ�Pandas���Կ��ٵĽ������ڡ� 

��ע��Pandas��read_csv()����Ҳ�ṩ�˽���ʱ��Ĳ��������parse_dates��infer_datetime_format����date_parser������ 

# ����
����ʱ���Ѿ���ɸ�ʽ��������׼����ʼ�������ˡ�����ÿ��ʱ�εĵ�۲�ͬ�������Ҫ����Ӧ�ĵ��ӳ�䵽����ʱ�Ρ������У�����շѱ�׼���£�
![testpic](https://github.com/monkeyshichi/python_code_optimize/blob/master/python_speedup_codesample/imgfolder/tb2.png)

������ȫ��ͳһ��28����ÿǧ��ÿСʱ��������˶�֪������һ�д���ʵ�ֵ�ѵļ��㣺
```
>>> df['cost_cents'] = df['energy_kwh'] * 28
```
���д��뽫����һ�����У����а�����ǰʱ�εĵ�ѣ�

```
   date_time    energy_kwh       cost_cents
0    2013-01-01 00:00:00         0.586           16.408
1    2013-01-01 01:00:00         0.580           16.240
2    2013-01-01 02:00:00         0.572           16.016
3    2013-01-01 03:00:00         0.596           16.688
4    2013-01-01 04:00:00         0.592           16.576
# ...
```

���ǵ�ѵļ���ȡ���ڲ��õ�ʱ�ζ�Ӧ�ĵ�ۡ���������˻��÷�Pandasʽ�ķ�ʽ���ñ���ȥ���������㡣 

�ڱ����У�����������Ľ��������ʼ���ܣ������ṩ�������Pandas�������Ƶ�Pythonʽ��������� 

���Ƕ���Pandas����˵��ʲô��Pythonʽ������������ָ��������Ѻ��Խϲ��������C++����Java�������Ѿ�ϰ���ˡ����ñ�����ȥ��̡� 

�������ϤPandas��������˻�����ǰһ��ʹ�ü��������������������ʹ��@timeitװ�������������ַ�����Ч�ʡ� 

���ȣ�����һ����ͬʱ�ε�۵ĺ�����
```
def apply_tariff(kwh, hour):
    """��ۺ���"""    
    if 0 <= hour < 7:
        rate = 12
    elif 7 <= hour < 17:
        rate = 20
    elif 17 <= hour < 24:
        rate = 28
    else:
        raise ValueError(f'Invalid hour: {hour}')
    return rate * kwh
```	
���´������һ�ֳ����ı���ģʽ��
```
>>> # ע�⣺��Ҫ���Ըú�����
>>> @timeit(repeat=3, number=100)
... def apply_tariff_loop(df):
...     """�ñ�������ɱ�����������뵽df��"""
...     energy_cost_list = []
...     for i in range(len(df)):
...         # ��ȡÿ��Сʱ�ĺĵ���
...         energy_used = df.iloc[i]['energy_kwh']
...         hour = df.iloc[i]['date_time'].hour
...         energy_cost = apply_tariff(energy_used, hour)
...         energy_cost_list.append(energy_cost)
...     df['cost_cents'] = energy_cost_list
... 
>>> apply_tariff_loop(df)
Best of 3 trials with 100 function calls per trial:
Function `apply_tariff_loop` ran in average of 3.152 seconds.
```

����û���ù�Pandas��Python�û���˵�����ֱ���������������ÿ��x���ٸ�������y�£����z��
 
�������ֱ����ܱ��ء����Խ�����������ΪPandas�÷��ġ����永������ԭ�����¼�����

���ȣ�����Ҫ��ʼ��һ���б����ڴ洢��������

��Σ������������Ѷ�����range(0, len(df))ȥ��ѭ����������Ӧ��apply_tariff()�����󣬻����뽫������ӵ��б������������µ�DataFrame�С�

�������ʹ����ʽ����df.iloc[i]['date_time']������ܻ��������ܶ�bugs�� 

���ֱ�����ʽ�����������ڼ����ʱ��ɱ�������8760�����ݣ�����3������ɱ��������棬������һЩ����Pandas���ݽṹ�ĵ���������

# ��.itertuples()��.iterrow()����

���������취��

Pandasʵ����ͨ������DataFrame.itertuples()��DataFrame.iterrows()����ʹ��for i in range(len(df))�﷨���ࡣ�����ֶ��ǲ���һ��һ�е������������� 

.itertuples()Ϊÿ������һ��nametuple�࣬�е�����ֵ��Ϊnametuple��ĵ�һ��Ԫ�ء�nametuple������Python��collectionsģ������ݽṹ���ýṹ��Python�е�Ԫ�����ƣ����ǿ���ͨ�����Բ��ҿɷ����ֶΡ� 

.iterrows()ΪDataFrame��ÿ������һ����������������ɵ�Ԫ�顣 

��.iterrows()��ȣ�.itertuples()�����ٶȻ����һЩ��������ʹ����.iterrows()��������Ϊ�ܶ���ߺܿ���û���ù�nametuple��

```
>>> @timeit(repeat=3, number=100)
... def apply_tariff_iterrows(df):
...     energy_cost_list = []
...     for index, row in df.iterrows():
...         #��ȡÿ��Сʱ�ĺĵ���
...         energy_used = row['energy_kwh']
...         hour = row['date_time'].hour
...         # ���ӳɱ����ݵ�list�б�
...         energy_cost = apply_tariff(energy_used, hour)
...         energy_cost_list.append(energy_cost)
...     df['cost_cents'] = energy_cost_list
...
>>> apply_tariff_iterrows(df)
Best of 3 trials with 100 function calls per trial:
Function `apply_tariff_iterrows` ran in average of 0.713 seconds.
```
ȡ��һЩ����Ľ������﷨��������������ֵi�����ã���������пɶ����ˡ���ʱ�����淽�棬��������5���� 

���ǣ���Ȼ�кܴ�ĸĽ��ռ䡣������Ȼ��ʹ��for��������ζ��ÿѭ��һ�ζ���Ҫ����һ�κ���������Щ���������ٶȸ����Pandas���üܹ�����ɡ�

# Pandas��.apply()
������.apply()�������.iterrows()��������Ч�ʡ�Pandas��.apply()�������Դ���ɵ��õĺ�������Ӧ����DataFrame�����ϣ��������л��С������У�����lambda���ܽ��������ݴ���apply_tariff()��

```
>>> @timeit(repeat=3, number=100)
... def apply_tariff_withapply(df):
...     df['cost_cents'] = df.apply(
...         lambda row: apply_tariff(
...             kwh=row['energy_kwh'],
...             hour=row['date_time'].hour),
...         axis=1)
...
>>> apply_tariff_withapply(df)
Best of 3 trials with 100 function calls per trial:
Function `apply_tariff_withapply` ran in average of 0.272 seconds.
```

.apply()���﷨���ƺ����ԣ������������ˣ�ͬʱ����Ҳ���׶��ˡ������ٶȷ��棬����.iterrows()������Ƚ�ʡ�˴�Լһ��ʱ�䡣 

���ǣ��⻹�����졣һ��ԭ����.apply()�ڲ�������Cython�����������ѭ������������������£�lambda�д�����һЩ�޷���Cython�д�������룬��˵���.apply()ʱ��Ȼ�����졣 

���ʹ��.apply()��330��վ���10�������ϣ����ŵû�15���ӵĴ���ʱ�䡣����������������һ������ģ�͵�һС���֣���ô����Ҫ������������������������������������㡣

# ��.isin()ɸѡ����
֮ǰ���������ֻ�е�һ��ۣ����Խ����е����������ݳ��Ըü۸�df['energy_kwh'] * 28�����ֲ�������һ��������������һ������������Pandas�����ķ�ʽ�� 

���ǣ���Pandas����ν��������ļ���Ӧ�����������������أ�һ�ַ����ǣ�����������DataFrame����ɸѡ���������Ƭ��Ȼ���ÿ�����ݽ��ж�Ӧ�������������� 

������������У���չʾ���ʹ��Pandas�е�.isin()����ɸѡ�У�Ȼ�������������������Ӧ�ĵ�ѡ��ڴ˲���ǰ����date_time������ΪDataFrame��������������������

```
df.set_index('date_time', inplace=True)

@timeit(repeat=3, number=100)
def apply_tariff_isin(df):
    # ����ÿ��ʱ�εĲ���������(Boolean)
    peak_hours = df.index.hour.isin(range(17, 24))
    shoulder_hours = df.index.hour.isin(range(7, 17))
    off_peak_hours = df.index.hour.isin(range(0, 7)) 

    # ���㲻ͬʱ�εĵ��
    df.loc[peak_hours, 'cost_cents'] = df.loc[peak_hours, 'energy_kwh'] * 28
    df.loc[shoulder_hours,'cost_cents'] = df.loc[shoulder_hours, 'energy_kwh'] * 20
    df.loc[off_peak_hours,'cost_cents'] = df.loc[off_peak_hours, 'energy_kwh'] * 12
```
ִ�н�����£�
```
>>> apply_tariff_isin(df)
Best of 3 trials with 100 function calls per trial:
Function `apply_tariff_isin` ran in average of 0.010 seconds.
```

Ҫ�����δ��룬Ҳ����Ҫ���˽�.isin()�������ص��ǲ���ֵ�����£�

```
[False, False, False, ..., True, True, True]
```
��Щ����ֵ�����DataFrame����ʱ���������ڵ�ʱ�Ρ�Ȼ�󣬽���Щ����ֵ���鴫��DataFrame��.loc������ʱ���᷵��һ����������ʱ�ε�DataFrame��Ƭ����󣬽�����Ƭ������Զ�Ӧ��ʱ�εķ��ʼ��ɡ� 

����֮ǰ�ı������������Σ�

���ȣ�����Ҫapply_tariff()�����ˣ���Ϊ���е������߼�����Ӧ�����˱�ѡ�е��С���������˴����������

���ٶȷ��棬����ͨ�ı�������315������.iterrows()��������71�����ұ�.apply()��������27�������ڿ��Կ��ٵĴ�������ݼ��ˡ�

# ���������ռ���

��apply_tariff_isin()�У���Ҫ�ֶ���������df.loc��df.index.hour.isin()������24Сʱÿ��Сʱ�ķ��ʲ�ͬ������ζ����Ҫ�ֶ�����24��.isin()�������������ַ���ͨ����������չ�ԡ����˵��ǣ�������ʹ��Pandas��pd.cut()���ܣ� 
```
@timeit(repeat=3, number=100)
def apply_tariff_cut(df):
    cents_per_kwh = pd.cut(x=df.index.hour,
                           bins=[0, 7, 17, 24],
                           include_lowest=True,
                           labels=[12, 20, 28]).astype(int)
    df['cost_cents'] = cents_per_kwh * df['energy_kwh']
```

pd.cut()���ݷ���bins�������������ɶ�Ӧ�ı�ǩ�����ʡ���

��ע��include_lowest�����趨��һ������Ƿ��������bins�У�������Ҫ�ڸ����а���ʱ����0ʱ������ݡ�
 
��������ȫ�������Ĳ���������ִ���ٶ��Ѿ�����ˣ� 
```
>>> apply_tariff_cut(df)
Best of 3 trials with 100 function calls per trial:
Function `apply_tariff_cut` ran in average of 0.003 seconds.
```
���ˣ����ڿ��Խ�330��վ������ݴ���ʱ���88������С��ֻ�費��1�롣���ǣ��������һ��ѡ�񣬾���ʹ��NumPy��������DataFrame�µ�ÿ��NumPy���飬Ȼ�󽫴��������ɻ�DataFrame���ݽṹ�С�

# ����NumPy��

������Pandas��Series��DataFrame����NumPy��Ļ�������Ƶġ����ṩ�˸��������ԣ���ΪPandas��NumPy��������޷������ 

����һ���У�����ʾNumPy��digitize()���ܡ�����Pandas��cut()�������ƣ������ݷ��顣�����н�DataFrame�е�����������ʱ�䣩���з��飬������ʱ�η������顣Ȼ�󽫷����ĵ�����������Ӧ���ڵ�������ϣ� 

```
@timeit(repeat=3, number=100)
def apply_tariff_digitize(df):
    prices = np.array([12, 20, 28])
    bins = np.digitize(df.index.hour.values, bins=[7, 17, 24])
    df['cost_cents'] = prices[bins] * df['energy_kwh'].values
```

��cut()һ�����﷨���׶��������ٶ�����أ� 

```
>>> apply_tariff_digitize(df)
Best of 3 trials with 100 function calls per trial:
Function `apply_tariff_digitize` ran in average of 0.002 seconds.
```

ִ���ٶ��ϣ���Ȼ���������������������Ѿ����岻���ˡ����罫���ྫ��ȥ˼�����������顣
 
Pandas�����ṩ�ܶ������������ݷ����ı���ѡ���Щ�Ѿ����ϱ߶�һһ��ʾ���ˡ����ｫ��쵽�����ķ����������£�
 
1. ʹ��������������û��for������Pandas�����ͺ�����

2. ʹ��.apply()������

3. ʹ��.itertuples()����DataFrame����Ϊnametuple���Python��collectionsģ���н��е�����

4. ʹ��.iterrows()����DataFrame����Ϊ��index��pd.Series��Ԫ��������е�������ȻPandas��Series��һ���������ݽṹ������ÿһ������һ��Series���ҷ���������Ȼ��һ���Ƚϴ�Ŀ�����

5. �����Ԫ�ؽ���ѭ����ʹ��df.loc����df.iloc��ÿ����Ԫ������н��д���

��ע������˳�������ߵĽ��飬����Pandas���Ŀ�����Ա���Ľ��顣
 
�����Ǳ��������к����ĵ���ʱ����ܣ�

![testpic](https://github.com/monkeyshichi/python_code_optimize/blob/master/python_speedup_codesample/imgfolder/tb3.png)

# ��HDFstore�洢Ԥ��������

�Ѿ��˽�����Pandas���ٴ������ݣ�����������Ҫ̽����α����ظ������ݴ�����̡�����ʹ����Pandas���õ�HDFStore������ 

ͨ���ڽ���һЩ���ӵ�����ģ��ʱ����������һЩԤ�����Ǻܳ����ġ����磬������10��ʱ���ȵķ��Ӽ��ĸ�Ƶ���ݣ�����ģ��ֻ��Ҫ20����Ƶ�ε����ݻ���������Ƶ�����ݡ��㲻ϣ��ÿ�β��Է���ģ��ʱ����ҪԤ�������ݡ� 

һ�ַ����ǣ����Ѿ����Ԥ��������ݴ洢���Ѵ������ݱ��У�������Ҫʱ��ʱ���á������������ȷ�ĸ�ʽ�洢���ݣ������Ԥ�����������ΪCSV����ô�ᶪʧdatetime�࣬�ٴζ���ʱ��������ת����ʽ�� 

Pandas�и����õĽ����������ʹ��HDF5������һ��ר�����ڴ洢����ĸ����ܴ洢��ʽ��Pandas��HDFstore�������Խ�DataFrame�洢��HDF5�ļ��У�������Ч��д��ͬʱ��Ȼ����DataFrame���е��������ͺ�����Ԫ���ݡ�����һ�������ֵ���࣬��˿�����Python�е�dict��һ����д�� 

�����ǽ��Ѿ�Ԥ����ĺĵ���DataFrameд��HDF5�ļ��ķ����� 

```
# �����洢���ļ������� `processed_data`
data_store = pd.HDFStore('processed_data.h5')

#��DataFrameд��洢�ļ��У������ü���key�� 'preprocessed_df'
data_store['preprocessed_df'] = df
data_store.close()
```

�����ݴ洢��Ӳ���Ժ󣬿�����ʱ��ص�ȡԤ�������ݣ�������Ҫ�ظ��ӹ��������ǹ�����δ�HDF5�ļ��з������ݵķ�����ͬʱ����������Ԥ����ʱ���������ͣ�

```
# �������ݲֿ�
data_store = pd.HDFStore('processed_data.h5')

# ��ȡ����key��Ϊ'preprocessed_df'��DataFrame
preprocessed_df = data_store['preprocessed_df']
data_store.close()
```

һ�����ݲֿ���Դ洢���ű�ÿ�ű�����һ������ 

��ע��ʹ��Pandas��HDFStore��Ҫ��װPyTables>=3.0.0����˰�װPandas����Ҫ����PyTables��

```
pip install --upgrade tables
```

# �ܽ�
	
����������Pandas��Ŀ���߱��ٶȿ졢������ֱ�۵���������ô������˼��ʹ�øÿ�ķ�ʽ�ˡ�
 
�������Ѿ��൱ֱ�۵�չʾ����ȷ��ʹ��Pandas�ǿ��Դ����������ʱ�䣬�Լ�����ɶ��Եġ�������Ӧ��Pandas��һЩ�����ԵĽ��飺
 
�� ���Ը����������������������������for x in df�Ĳ�������������б���������forѭ������ô����ʹ��Python�Դ������ݽṹ����ΪPandas������ܶ࿪����

�� �����Ϊ�㷨�����޷�ʹ�����������������Գ���.apply()������

�� �������ѭ���������飬����.iterrows()����.itertuples()�Ľ��﷨�������ٶȡ�

�� Pandas�кܶ��ѡ����������м��ַ���������ɴ�A��B�Ĺ��̣��Ƚϲ�ͬ������ִ�з�ʽ��ѡ�����ʺ���Ŀ��һ�֡�

�� �������ݴ���ű��󣬿��Խ��м������Ԥ�������ݱ�����HDFStore�У��������´������ݡ�

�� ��Pandas��Ŀ�У�����NumPy��������ٶ�ͬʱ���﷨��

���ο����ӡ�
https://github.com/realpython/materials/tree/master/pandas-fast-flexible-intuitive��1��
https://realpython.com/fast-flexible-pandas/