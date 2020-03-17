

```python
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

# 문제 정의

FIFA 19의 선수 스텟을 바탕으로, 그 선수의 포지션을 예측하라

# 데이터 수집 및 전처리


```python
# 데이터를 수집합니다
df = pd.read_csv("../data/fifa_data.csv")
```


```python
# 수집된 데이터 샘플을 확인합니다
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>Photo</th>
      <th>Nationality</th>
      <th>Flag</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>...</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
      <th>Release Clause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>158023</td>
      <td>L. Messi</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/158023.png</td>
      <td>Argentina</td>
      <td>https://cdn.sofifa.org/flags/52.png</td>
      <td>94</td>
      <td>94</td>
      <td>FC Barcelona</td>
      <td>...</td>
      <td>96.0</td>
      <td>33.0</td>
      <td>28.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>€226.5M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20801</td>
      <td>Cristiano Ronaldo</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/20801.png</td>
      <td>Portugal</td>
      <td>https://cdn.sofifa.org/flags/38.png</td>
      <td>94</td>
      <td>94</td>
      <td>Juventus</td>
      <td>...</td>
      <td>95.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>€127.1M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>190871</td>
      <td>Neymar Jr</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/190871.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>92</td>
      <td>93</td>
      <td>Paris Saint-Germain</td>
      <td>...</td>
      <td>94.0</td>
      <td>27.0</td>
      <td>24.0</td>
      <td>33.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>€228.1M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>193080</td>
      <td>De Gea</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/193080.png</td>
      <td>Spain</td>
      <td>https://cdn.sofifa.org/flags/45.png</td>
      <td>91</td>
      <td>93</td>
      <td>Manchester United</td>
      <td>...</td>
      <td>68.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>13.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>94.0</td>
      <td>€138.6M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>192985</td>
      <td>K. De Bruyne</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/192985.png</td>
      <td>Belgium</td>
      <td>https://cdn.sofifa.org/flags/7.png</td>
      <td>91</td>
      <td>92</td>
      <td>Manchester City</td>
      <td>...</td>
      <td>88.0</td>
      <td>68.0</td>
      <td>58.0</td>
      <td>51.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>€196.4M</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 89 columns</p>
</div>




```python
df1 = pd.DataFrame({'Name': df.Name,'Club':df.Club,'Position':df.Position,}) #선수정보데이터
df2 = df.iloc[:,54:88].astype(float) #스텟데이터

td = pd.concat([df1,df2],axis=1)
td
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Club</th>
      <th>Position</th>
      <th>Crossing</th>
      <th>Finishing</th>
      <th>HeadingAccuracy</th>
      <th>ShortPassing</th>
      <th>Volleys</th>
      <th>Dribbling</th>
      <th>Curve</th>
      <th>...</th>
      <th>Penalties</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>L. Messi</td>
      <td>FC Barcelona</td>
      <td>RF</td>
      <td>84.0</td>
      <td>95.0</td>
      <td>70.0</td>
      <td>90.0</td>
      <td>86.0</td>
      <td>97.0</td>
      <td>93.0</td>
      <td>...</td>
      <td>75.0</td>
      <td>96.0</td>
      <td>33.0</td>
      <td>28.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cristiano Ronaldo</td>
      <td>Juventus</td>
      <td>ST</td>
      <td>84.0</td>
      <td>94.0</td>
      <td>89.0</td>
      <td>81.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>81.0</td>
      <td>...</td>
      <td>85.0</td>
      <td>95.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Neymar Jr</td>
      <td>Paris Saint-Germain</td>
      <td>LW</td>
      <td>79.0</td>
      <td>87.0</td>
      <td>62.0</td>
      <td>84.0</td>
      <td>84.0</td>
      <td>96.0</td>
      <td>88.0</td>
      <td>...</td>
      <td>81.0</td>
      <td>94.0</td>
      <td>27.0</td>
      <td>24.0</td>
      <td>33.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>De Gea</td>
      <td>Manchester United</td>
      <td>GK</td>
      <td>17.0</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>50.0</td>
      <td>13.0</td>
      <td>18.0</td>
      <td>21.0</td>
      <td>...</td>
      <td>40.0</td>
      <td>68.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>13.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>K. De Bruyne</td>
      <td>Manchester City</td>
      <td>RCM</td>
      <td>93.0</td>
      <td>82.0</td>
      <td>55.0</td>
      <td>92.0</td>
      <td>82.0</td>
      <td>86.0</td>
      <td>85.0</td>
      <td>...</td>
      <td>79.0</td>
      <td>88.0</td>
      <td>68.0</td>
      <td>58.0</td>
      <td>51.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>E. Hazard</td>
      <td>Chelsea</td>
      <td>LF</td>
      <td>81.0</td>
      <td>84.0</td>
      <td>61.0</td>
      <td>89.0</td>
      <td>80.0</td>
      <td>95.0</td>
      <td>83.0</td>
      <td>...</td>
      <td>86.0</td>
      <td>91.0</td>
      <td>34.0</td>
      <td>27.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>L. Modrić</td>
      <td>Real Madrid</td>
      <td>RCM</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>55.0</td>
      <td>93.0</td>
      <td>76.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>...</td>
      <td>82.0</td>
      <td>84.0</td>
      <td>60.0</td>
      <td>76.0</td>
      <td>73.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>L. Suárez</td>
      <td>FC Barcelona</td>
      <td>RS</td>
      <td>77.0</td>
      <td>93.0</td>
      <td>77.0</td>
      <td>82.0</td>
      <td>88.0</td>
      <td>87.0</td>
      <td>86.0</td>
      <td>...</td>
      <td>85.0</td>
      <td>85.0</td>
      <td>62.0</td>
      <td>45.0</td>
      <td>38.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>31.0</td>
      <td>33.0</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sergio Ramos</td>
      <td>Real Madrid</td>
      <td>RCB</td>
      <td>66.0</td>
      <td>60.0</td>
      <td>91.0</td>
      <td>78.0</td>
      <td>66.0</td>
      <td>63.0</td>
      <td>74.0</td>
      <td>...</td>
      <td>75.0</td>
      <td>82.0</td>
      <td>87.0</td>
      <td>92.0</td>
      <td>91.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>J. Oblak</td>
      <td>Atlético Madrid</td>
      <td>GK</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>29.0</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>70.0</td>
      <td>27.0</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>86.0</td>
      <td>92.0</td>
      <td>78.0</td>
      <td>88.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>R. Lewandowski</td>
      <td>FC Bayern München</td>
      <td>ST</td>
      <td>62.0</td>
      <td>91.0</td>
      <td>85.0</td>
      <td>83.0</td>
      <td>89.0</td>
      <td>85.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>88.0</td>
      <td>86.0</td>
      <td>34.0</td>
      <td>42.0</td>
      <td>19.0</td>
      <td>15.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>T. Kroos</td>
      <td>Real Madrid</td>
      <td>LCM</td>
      <td>88.0</td>
      <td>76.0</td>
      <td>54.0</td>
      <td>92.0</td>
      <td>82.0</td>
      <td>81.0</td>
      <td>86.0</td>
      <td>...</td>
      <td>73.0</td>
      <td>85.0</td>
      <td>72.0</td>
      <td>79.0</td>
      <td>69.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>D. Godín</td>
      <td>Atlético Madrid</td>
      <td>CB</td>
      <td>55.0</td>
      <td>42.0</td>
      <td>92.0</td>
      <td>79.0</td>
      <td>47.0</td>
      <td>53.0</td>
      <td>49.0</td>
      <td>...</td>
      <td>50.0</td>
      <td>82.0</td>
      <td>90.0</td>
      <td>89.0</td>
      <td>89.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>15.0</td>
      <td>5.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>David Silva</td>
      <td>Manchester City</td>
      <td>LCM</td>
      <td>84.0</td>
      <td>76.0</td>
      <td>54.0</td>
      <td>93.0</td>
      <td>82.0</td>
      <td>89.0</td>
      <td>82.0</td>
      <td>...</td>
      <td>75.0</td>
      <td>93.0</td>
      <td>59.0</td>
      <td>53.0</td>
      <td>29.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>N. Kanté</td>
      <td>Chelsea</td>
      <td>LDM</td>
      <td>68.0</td>
      <td>65.0</td>
      <td>54.0</td>
      <td>86.0</td>
      <td>56.0</td>
      <td>79.0</td>
      <td>49.0</td>
      <td>...</td>
      <td>54.0</td>
      <td>85.0</td>
      <td>90.0</td>
      <td>91.0</td>
      <td>85.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>P. Dybala</td>
      <td>Juventus</td>
      <td>LF</td>
      <td>82.0</td>
      <td>84.0</td>
      <td>68.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>92.0</td>
      <td>88.0</td>
      <td>...</td>
      <td>86.0</td>
      <td>84.0</td>
      <td>23.0</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>H. Kane</td>
      <td>Tottenham Hotspur</td>
      <td>ST</td>
      <td>75.0</td>
      <td>94.0</td>
      <td>85.0</td>
      <td>80.0</td>
      <td>84.0</td>
      <td>80.0</td>
      <td>78.0</td>
      <td>...</td>
      <td>90.0</td>
      <td>89.0</td>
      <td>56.0</td>
      <td>36.0</td>
      <td>38.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>A. Griezmann</td>
      <td>Atlético Madrid</td>
      <td>CAM</td>
      <td>82.0</td>
      <td>90.0</td>
      <td>84.0</td>
      <td>83.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>84.0</td>
      <td>...</td>
      <td>79.0</td>
      <td>87.0</td>
      <td>59.0</td>
      <td>47.0</td>
      <td>48.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M. ter Stegen</td>
      <td>FC Barcelona</td>
      <td>GK</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>36.0</td>
      <td>14.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>...</td>
      <td>25.0</td>
      <td>69.0</td>
      <td>25.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>87.0</td>
      <td>85.0</td>
      <td>88.0</td>
      <td>85.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>T. Courtois</td>
      <td>Real Madrid</td>
      <td>GK</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>33.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>19.0</td>
      <td>...</td>
      <td>27.0</td>
      <td>66.0</td>
      <td>20.0</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>85.0</td>
      <td>91.0</td>
      <td>72.0</td>
      <td>86.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Sergio Busquets</td>
      <td>FC Barcelona</td>
      <td>CDM</td>
      <td>62.0</td>
      <td>67.0</td>
      <td>68.0</td>
      <td>89.0</td>
      <td>44.0</td>
      <td>80.0</td>
      <td>66.0</td>
      <td>...</td>
      <td>60.0</td>
      <td>90.0</td>
      <td>90.0</td>
      <td>86.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>E. Cavani</td>
      <td>Paris Saint-Germain</td>
      <td>LS</td>
      <td>70.0</td>
      <td>89.0</td>
      <td>89.0</td>
      <td>78.0</td>
      <td>90.0</td>
      <td>80.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>85.0</td>
      <td>82.0</td>
      <td>52.0</td>
      <td>45.0</td>
      <td>39.0</td>
      <td>12.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M. Neuer</td>
      <td>FC Bayern München</td>
      <td>GK</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>25.0</td>
      <td>55.0</td>
      <td>11.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>...</td>
      <td>47.0</td>
      <td>70.0</td>
      <td>17.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>90.0</td>
      <td>86.0</td>
      <td>91.0</td>
      <td>87.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>S. Agüero</td>
      <td>Manchester City</td>
      <td>ST</td>
      <td>70.0</td>
      <td>93.0</td>
      <td>77.0</td>
      <td>81.0</td>
      <td>85.0</td>
      <td>89.0</td>
      <td>82.0</td>
      <td>...</td>
      <td>83.0</td>
      <td>90.0</td>
      <td>30.0</td>
      <td>20.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>G. Chiellini</td>
      <td>Juventus</td>
      <td>LCB</td>
      <td>58.0</td>
      <td>33.0</td>
      <td>83.0</td>
      <td>59.0</td>
      <td>45.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>...</td>
      <td>50.0</td>
      <td>84.0</td>
      <td>93.0</td>
      <td>93.0</td>
      <td>90.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>K. Mbappé</td>
      <td>Paris Saint-Germain</td>
      <td>RM</td>
      <td>77.0</td>
      <td>88.0</td>
      <td>77.0</td>
      <td>82.0</td>
      <td>78.0</td>
      <td>90.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>70.0</td>
      <td>86.0</td>
      <td>34.0</td>
      <td>34.0</td>
      <td>32.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>M. Salah</td>
      <td>Liverpool</td>
      <td>RM</td>
      <td>78.0</td>
      <td>90.0</td>
      <td>59.0</td>
      <td>82.0</td>
      <td>73.0</td>
      <td>89.0</td>
      <td>83.0</td>
      <td>...</td>
      <td>61.0</td>
      <td>91.0</td>
      <td>38.0</td>
      <td>43.0</td>
      <td>41.0</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Casemiro</td>
      <td>Real Madrid</td>
      <td>CDM</td>
      <td>52.0</td>
      <td>59.0</td>
      <td>76.0</td>
      <td>85.0</td>
      <td>53.0</td>
      <td>69.0</td>
      <td>59.0</td>
      <td>...</td>
      <td>66.0</td>
      <td>84.0</td>
      <td>88.0</td>
      <td>90.0</td>
      <td>87.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>16.0</td>
      <td>12.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>J. Rodríguez</td>
      <td>FC Bayern München</td>
      <td>LAM</td>
      <td>90.0</td>
      <td>83.0</td>
      <td>62.0</td>
      <td>89.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>89.0</td>
      <td>...</td>
      <td>81.0</td>
      <td>87.0</td>
      <td>52.0</td>
      <td>41.0</td>
      <td>44.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>5.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>L. Insigne</td>
      <td>Napoli</td>
      <td>LW</td>
      <td>86.0</td>
      <td>77.0</td>
      <td>56.0</td>
      <td>85.0</td>
      <td>74.0</td>
      <td>90.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>61.0</td>
      <td>83.0</td>
      <td>51.0</td>
      <td>24.0</td>
      <td>22.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18177</th>
      <td>R. Roache</td>
      <td>Blackpool</td>
      <td>ST</td>
      <td>32.0</td>
      <td>53.0</td>
      <td>45.0</td>
      <td>37.0</td>
      <td>35.0</td>
      <td>44.0</td>
      <td>39.0</td>
      <td>...</td>
      <td>55.0</td>
      <td>49.0</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>18178</th>
      <td>L. Wahlstedt</td>
      <td>Dalkurd FF</td>
      <td>GK</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>...</td>
      <td>12.0</td>
      <td>28.0</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>47.0</td>
      <td>46.0</td>
      <td>50.0</td>
      <td>45.0</td>
      <td>51.0</td>
    </tr>
    <tr>
      <th>18179</th>
      <td>J. Williams</td>
      <td>Northampton Town</td>
      <td>CB</td>
      <td>25.0</td>
      <td>22.0</td>
      <td>43.0</td>
      <td>30.0</td>
      <td>29.0</td>
      <td>22.0</td>
      <td>27.0</td>
      <td>...</td>
      <td>32.0</td>
      <td>37.0</td>
      <td>42.0</td>
      <td>51.0</td>
      <td>49.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>18180</th>
      <td>M. Hurst</td>
      <td>St. Johnstone FC</td>
      <td>GK</td>
      <td>19.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>26.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>...</td>
      <td>29.0</td>
      <td>28.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>16.0</td>
      <td>45.0</td>
      <td>49.0</td>
      <td>50.0</td>
      <td>50.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>18181</th>
      <td>C. Maher</td>
      <td>Bray Wanderers</td>
      <td>RB</td>
      <td>45.0</td>
      <td>46.0</td>
      <td>46.0</td>
      <td>38.0</td>
      <td>27.0</td>
      <td>46.0</td>
      <td>28.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>43.0</td>
      <td>49.0</td>
      <td>45.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>18182</th>
      <td>Y. Góez</td>
      <td>Atlético Nacional</td>
      <td>CDM</td>
      <td>35.0</td>
      <td>33.0</td>
      <td>44.0</td>
      <td>55.0</td>
      <td>27.0</td>
      <td>42.0</td>
      <td>37.0</td>
      <td>...</td>
      <td>44.0</td>
      <td>38.0</td>
      <td>44.0</td>
      <td>42.0</td>
      <td>46.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>18183</th>
      <td>K. Pilkington</td>
      <td>Cambridge United</td>
      <td>GK</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>...</td>
      <td>22.0</td>
      <td>56.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>45.0</td>
      <td>48.0</td>
      <td>44.0</td>
      <td>49.0</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>18184</th>
      <td>D. Horton</td>
      <td>Lincoln City</td>
      <td>CM</td>
      <td>33.0</td>
      <td>24.0</td>
      <td>42.0</td>
      <td>54.0</td>
      <td>33.0</td>
      <td>44.0</td>
      <td>34.0</td>
      <td>...</td>
      <td>37.0</td>
      <td>42.0</td>
      <td>47.0</td>
      <td>49.0</td>
      <td>53.0</td>
      <td>12.0</td>
      <td>5.0</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>18185</th>
      <td>E. Tweed</td>
      <td>Derry City</td>
      <td>LCM</td>
      <td>37.0</td>
      <td>34.0</td>
      <td>49.0</td>
      <td>55.0</td>
      <td>27.0</td>
      <td>40.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>40.0</td>
      <td>43.0</td>
      <td>39.0</td>
      <td>39.0</td>
      <td>48.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>18186</th>
      <td>Zhang Yufeng</td>
      <td>Beijing Renhe FC</td>
      <td>CM</td>
      <td>35.0</td>
      <td>29.0</td>
      <td>47.0</td>
      <td>53.0</td>
      <td>33.0</td>
      <td>47.0</td>
      <td>38.0</td>
      <td>...</td>
      <td>43.0</td>
      <td>39.0</td>
      <td>53.0</td>
      <td>41.0</td>
      <td>51.0</td>
      <td>15.0</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>18187</th>
      <td>C. Ehlich</td>
      <td>SpVgg Unterhaching</td>
      <td>RB</td>
      <td>39.0</td>
      <td>40.0</td>
      <td>45.0</td>
      <td>46.0</td>
      <td>42.0</td>
      <td>46.0</td>
      <td>35.0</td>
      <td>...</td>
      <td>47.0</td>
      <td>47.0</td>
      <td>40.0</td>
      <td>42.0</td>
      <td>42.0</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>18188</th>
      <td>L. Collins</td>
      <td>Newport County</td>
      <td>CM</td>
      <td>41.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>50.0</td>
      <td>33.0</td>
      <td>51.0</td>
      <td>41.0</td>
      <td>...</td>
      <td>36.0</td>
      <td>46.0</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>41.0</td>
      <td>5.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>18189</th>
      <td>A. Kaltner</td>
      <td>SpVgg Unterhaching</td>
      <td>ST</td>
      <td>37.0</td>
      <td>48.0</td>
      <td>30.0</td>
      <td>45.0</td>
      <td>43.0</td>
      <td>50.0</td>
      <td>41.0</td>
      <td>...</td>
      <td>48.0</td>
      <td>37.0</td>
      <td>28.0</td>
      <td>15.0</td>
      <td>22.0</td>
      <td>15.0</td>
      <td>5.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>18190</th>
      <td>L. Watkins</td>
      <td>Cambridge United</td>
      <td>CM</td>
      <td>32.0</td>
      <td>32.0</td>
      <td>45.0</td>
      <td>48.0</td>
      <td>31.0</td>
      <td>41.0</td>
      <td>32.0</td>
      <td>...</td>
      <td>34.0</td>
      <td>46.0</td>
      <td>35.0</td>
      <td>44.0</td>
      <td>47.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>10.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>18191</th>
      <td>J. Norville-Williams</td>
      <td>Cambridge United</td>
      <td>LB</td>
      <td>47.0</td>
      <td>26.0</td>
      <td>39.0</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>45.0</td>
      <td>29.0</td>
      <td>...</td>
      <td>29.0</td>
      <td>36.0</td>
      <td>45.0</td>
      <td>42.0</td>
      <td>46.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>18192</th>
      <td>S. Squire</td>
      <td>Cambridge United</td>
      <td>CDM</td>
      <td>39.0</td>
      <td>36.0</td>
      <td>48.0</td>
      <td>46.0</td>
      <td>26.0</td>
      <td>44.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>41.0</td>
      <td>41.0</td>
      <td>44.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>18193</th>
      <td>N. Fuentes</td>
      <td>Unión Española</td>
      <td>RB</td>
      <td>36.0</td>
      <td>25.0</td>
      <td>40.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>46.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>32.0</td>
      <td>32.0</td>
      <td>41.0</td>
      <td>48.0</td>
      <td>48.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>18194</th>
      <td>J. Milli</td>
      <td>Lecce</td>
      <td>GK</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>52.0</td>
      <td>52.0</td>
      <td>52.0</td>
      <td>40.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>18195</th>
      <td>S. Griffin</td>
      <td>Waterford FC</td>
      <td>CM</td>
      <td>35.0</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>53.0</td>
      <td>36.0</td>
      <td>41.0</td>
      <td>36.0</td>
      <td>...</td>
      <td>35.0</td>
      <td>41.0</td>
      <td>44.0</td>
      <td>37.0</td>
      <td>48.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>7.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>18196</th>
      <td>K. Fujikawa</td>
      <td>Júbilo Iwata</td>
      <td>CM</td>
      <td>31.0</td>
      <td>28.0</td>
      <td>40.0</td>
      <td>53.0</td>
      <td>31.0</td>
      <td>46.0</td>
      <td>39.0</td>
      <td>...</td>
      <td>44.0</td>
      <td>35.0</td>
      <td>41.0</td>
      <td>44.0</td>
      <td>54.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>18197</th>
      <td>D. Holland</td>
      <td>Cork City</td>
      <td>CM</td>
      <td>44.0</td>
      <td>44.0</td>
      <td>36.0</td>
      <td>53.0</td>
      <td>43.0</td>
      <td>50.0</td>
      <td>48.0</td>
      <td>...</td>
      <td>49.0</td>
      <td>52.0</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>38.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>18198</th>
      <td>J. Livesey</td>
      <td>Burton Albion</td>
      <td>GK</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>19.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>14.0</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>46.0</td>
      <td>52.0</td>
      <td>58.0</td>
      <td>42.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>18199</th>
      <td>M. Baldisimo</td>
      <td>Vancouver Whitecaps FC</td>
      <td>CM</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>41.0</td>
      <td>51.0</td>
      <td>26.0</td>
      <td>46.0</td>
      <td>35.0</td>
      <td>...</td>
      <td>36.0</td>
      <td>40.0</td>
      <td>48.0</td>
      <td>49.0</td>
      <td>49.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>18200</th>
      <td>J. Young</td>
      <td>Swindon Town</td>
      <td>ST</td>
      <td>28.0</td>
      <td>47.0</td>
      <td>47.0</td>
      <td>42.0</td>
      <td>37.0</td>
      <td>39.0</td>
      <td>32.0</td>
      <td>...</td>
      <td>58.0</td>
      <td>50.0</td>
      <td>15.0</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>18201</th>
      <td>D. Walsh</td>
      <td>Waterford FC</td>
      <td>RB</td>
      <td>22.0</td>
      <td>23.0</td>
      <td>45.0</td>
      <td>25.0</td>
      <td>27.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>...</td>
      <td>38.0</td>
      <td>43.0</td>
      <td>44.0</td>
      <td>47.0</td>
      <td>53.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>18202</th>
      <td>J. Lundstram</td>
      <td>Crewe Alexandra</td>
      <td>CM</td>
      <td>34.0</td>
      <td>38.0</td>
      <td>40.0</td>
      <td>49.0</td>
      <td>25.0</td>
      <td>42.0</td>
      <td>30.0</td>
      <td>...</td>
      <td>43.0</td>
      <td>45.0</td>
      <td>40.0</td>
      <td>48.0</td>
      <td>47.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>18203</th>
      <td>N. Christoffersson</td>
      <td>Trelleborgs FF</td>
      <td>ST</td>
      <td>23.0</td>
      <td>52.0</td>
      <td>52.0</td>
      <td>43.0</td>
      <td>36.0</td>
      <td>39.0</td>
      <td>32.0</td>
      <td>...</td>
      <td>43.0</td>
      <td>42.0</td>
      <td>22.0</td>
      <td>15.0</td>
      <td>19.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>18204</th>
      <td>B. Worman</td>
      <td>Cambridge United</td>
      <td>ST</td>
      <td>25.0</td>
      <td>40.0</td>
      <td>46.0</td>
      <td>38.0</td>
      <td>38.0</td>
      <td>45.0</td>
      <td>38.0</td>
      <td>...</td>
      <td>55.0</td>
      <td>41.0</td>
      <td>32.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>18205</th>
      <td>D. Walker-Rice</td>
      <td>Tranmere Rovers</td>
      <td>RW</td>
      <td>44.0</td>
      <td>50.0</td>
      <td>39.0</td>
      <td>42.0</td>
      <td>40.0</td>
      <td>51.0</td>
      <td>34.0</td>
      <td>...</td>
      <td>50.0</td>
      <td>46.0</td>
      <td>20.0</td>
      <td>25.0</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>18206</th>
      <td>G. Nugent</td>
      <td>Tranmere Rovers</td>
      <td>CM</td>
      <td>41.0</td>
      <td>34.0</td>
      <td>46.0</td>
      <td>48.0</td>
      <td>30.0</td>
      <td>43.0</td>
      <td>40.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>43.0</td>
      <td>40.0</td>
      <td>43.0</td>
      <td>50.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
<p>18207 rows × 37 columns</p>
</div>




```python
tmp = td.columns
```

* 분류에 필요없는 스텟을 제거한다.


```python
del td[tmp[9]]
del td[tmp[10]]
del td[tmp[11]]
del td[tmp[15]]
del td[tmp[16]]
del td[tmp[17]]
del td[tmp[18]]
del td[tmp[19]]
del td[tmp[20]]
del td[tmp[21]]
del td[tmp[22]]
del td[tmp[23]]
del td[tmp[28]]
td.columns
```




    Index(['Name', 'Club', 'Position', 'Crossing', 'Finishing', 'HeadingAccuracy',
           'ShortPassing', 'Volleys', 'Dribbling', 'BallControl', 'Acceleration',
           'SprintSpeed', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
           'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
           'GKKicking', 'GKPositioning', 'GKReflexes'],
          dtype='object')




```python
# 현재 가지고 있는 데이터에서, 포지션의 갯수를 확인한다
td.Position.value_counts()
```




    ST     2152
    GK     2025
    CB     1778
    CM     1394
    LB     1322
    RB     1291
    RM     1124
    LM     1095
    CAM     958
    CDM     948
    RCB     662
    LCB     648
    LCM     395
    RCM     391
    LW      381
    RW      370
    RDM     248
    LDM     243
    LS      207
    RS      203
    RWB      87
    LWB      78
    CF       74
    LAM      21
    RAM      21
    RF       16
    LF       15
    Name: Position, dtype: int64



* 비슷한 역할을 하는 포지션끼리는 합쳐서 포지션별 데이터의 수를 늘린다.


```python
td.loc[td['Position']=='LF', ['Position']] = 'ST'
td.loc[td['Position']=='RF', ['Position']] = 'ST'
td.loc[td['Position']=='CF', ['Position']] = 'ST'
td.loc[td['Position']=='LS', ['Position']] = 'ST'
td.loc[td['Position']=='RS', ['Position']] = 'ST'
td.loc[td['Position']=='LAM', ['Position']] = 'CAM'
td.loc[td['Position']=='RAM', ['Position']] = 'CAM'
td.loc[td['Position']=='LCM', ['Position']] = 'CM'
td.loc[td['Position']=='RCM', ['Position']] = 'CM'
td.loc[td['Position']=='RDM', ['Position']] = 'CDM'
td.loc[td['Position']=='LDM', ['Position']] = 'CDM'
td.loc[td['Position']=='LW', ['Position']] = 'WF'
td.loc[td['Position']=='RW', ['Position']] = 'WF'
td.loc[td['Position']=='LB', ['Position']] = 'WB'
td.loc[td['Position']=='RB', ['Position']] = 'WB'
td.loc[td['Position']=='LWB', ['Position']] = 'WB'
td.loc[td['Position']=='RWB', ['Position']] = 'WB'
td.loc[td['Position']=='LM', ['Position']] = 'WM'
td.loc[td['Position']=='RM', ['Position']] = 'WM'
td.loc[td['Position']=='LCB', ['Position']] = 'CB'
td.loc[td['Position']=='RCB', ['Position']] = 'CB'
```


```python
# 현재 가지고 있는 데이터에서, 포지션의 갯수를 확인한다
td.Position.value_counts()
```




    CB     3088
    WB     2778
    ST     2667
    WM     2219
    CM     2180
    GK     2025
    CDM    1439
    CAM    1000
    WF      751
    Name: Position, dtype: int64




```python
td = td.dropna() #null값 제거
```

# 데이터 나누기 (학습 데이터, 테스트 데이터)


```python
# sklearn의 train_test_split을 사용하면 라인 한줄로 손쉽게 데이터를 나눌 수 있다
from sklearn.model_selection import train_test_split

# 다듬어진 데이터에서 20%를 테스트 데이터로 분류합니다
train, test = train_test_split(td, test_size=0.2)
```


```python
# 학습 데이터의 갯수를 확인합니다, 14565개의 데이터가 있습니다.
train.shape[0]
```




    14334




```python
# 테스트 데이터의 갯수를 확인합니다. 3642개의 데이터가 있습니다.
test.shape[0]
```




    3584



# 다듬어진 데이터를 파일로 저장하기
다듬어진 데이터를 파일로 저장하여, 머신러닝 분류 알고리즘 실습 시에 사용하도록 하겠습니다.


```python
with open('../data/fifa_train.pkl', 'wb') as train_data:
    pickle.dump(train, train_data)
    
with open('../data/fifa_test.pkl', 'wb') as test_data:
    pickle.dump(test, test_data)
```

# 데이터 불러오기 (학습 데이터, 테스트 데이터)
학습 데이터 및 테스트 데이터를 로드합니다.


```python
with open('../data/fifa_train.pkl', 'rb') as train_data:
    train = pickle.load(train_data)
    
with open('../data/fifa_test.pkl', 'rb') as test_data:
    test = pickle.load(test_data)
```

## 2. SVM 최적의 파라미터 찾기
SVM의 파라미터는 두가지가 있습니다.
1. C : 비용 (cost), 결정경계선의 마진을 결정하는 파라미터입니다.
2. gamma: 커널의 데이터포인트의 표준편차를 결정하는 파라미터입니다.

결과적으로 C가 클수록, 결정경계선과 서포트 벡터의 간격(마진)이 작아집니다.  
결과적으로 gamma가 클수록, 결정경계선이 데이터포인트와 더욱 가까워집니다.


```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import numpy as np
```

sklearn에서 제공하는 gridsearch를 사용하시면, 손쉽게 최적의 C, gamma를 구하실 수 있습니다.


```python
def svc_param_selection(X, y, nfolds):
    svm_parameters = [
                        {'kernel': ['rbf'],
                         'gamma': [0.00001,0.0001, 0.001, 0.01, 0.1, 1],
                         'C': [0.01, 0.1, 1, 10, 100, 1000]
                        }
                       ]
    
    clf = GridSearchCV(SVC(), svm_parameters, cv=10)
    clf.fit(X_train, y_train.values.ravel())
    print(clf.best_params_)
    
    return clf
```


```python
X_train = train.iloc[:,3:24]
y_train = train[['Position']]
# 최적의 파라미터를 sklearn의 gridsearch를 통해 구합니다.
clf = svc_param_selection(X_train, y_train.values.ravel(), 10)
```

    {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
    


```python
clf
```




    GridSearchCV(cv=10, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid=[{'kernel': ['rbf'], 'gamma': [1e-05, 0.0001, 0.001, 0.01, 0.1, 1], 'C': [0.01, 0.1, 1, 10, 100, 1000]}],
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)



## 2. SVM
sklearn의 gridsearch로 얻어진 최적의 파라미터로 학습된 clf를 이용하여,  
테스트를 진행합니다.


```python
# 테스트에 사용될 특징을 지정합니다
X_test = test.iloc[:,3:24]

# 특징으로 예측할 값 (포지션)을 지정합니다
y_test = test[['Position']]

# 최적의 파라미터로 완성된 SVM에 테스트 데이터를 주입하여, 실제값과 예측값을 얻습니다.
y_true, y_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred))
print()
print("accuracy : "+ str(accuracy_score(y_true, y_pred)) )
```

                 precision    recall  f1-score   support
    
            CAM       0.52      0.46      0.49       188
             CB       0.87      0.87      0.87       602
            CDM       0.53      0.44      0.48       258
             CM       0.62      0.75      0.68       439
             GK       1.00      1.00      1.00       399
             ST       0.87      0.88      0.88       521
             WB       0.82      0.87      0.85       607
             WF       0.00      0.00      0.00       151
             WM       0.55      0.66      0.60       419
    
    avg / total       0.73      0.76      0.74      3584
    
    
    accuracy : 0.7575334821428571
    

    C:\Users\study\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    


```python
# 실제값(ground truth)과 예측값(prediction)이 어느 정도 일치하는 눈으로 직접 비교해봅니다
comparison = pd.DataFrame({'prediction':y_pred, 'ground_truth':y_true.values.ravel()}) 
comparison
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
      <th>ground_truth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WB</td>
      <td>WB</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GK</td>
      <td>GK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CM</td>
      <td>CAM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WB</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CB</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GK</td>
      <td>GK</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CDM</td>
      <td>CDM</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ST</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CAM</td>
      <td>WM</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CB</td>
      <td>WB</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CDM</td>
      <td>CDM</td>
    </tr>
    <tr>
      <th>11</th>
      <td>WB</td>
      <td>WB</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CB</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ST</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CB</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CM</td>
      <td>CM</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CM</td>
      <td>CM</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CM</td>
      <td>CM</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CAM</td>
      <td>CAM</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CM</td>
      <td>CM</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ST</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>21</th>
      <td>WB</td>
      <td>WB</td>
    </tr>
    <tr>
      <th>22</th>
      <td>WB</td>
      <td>WB</td>
    </tr>
    <tr>
      <th>23</th>
      <td>WB</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>24</th>
      <td>CM</td>
      <td>CM</td>
    </tr>
    <tr>
      <th>25</th>
      <td>CDM</td>
      <td>CDM</td>
    </tr>
    <tr>
      <th>26</th>
      <td>GK</td>
      <td>GK</td>
    </tr>
    <tr>
      <th>27</th>
      <td>WM</td>
      <td>WM</td>
    </tr>
    <tr>
      <th>28</th>
      <td>WB</td>
      <td>CM</td>
    </tr>
    <tr>
      <th>29</th>
      <td>CM</td>
      <td>CAM</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3554</th>
      <td>CAM</td>
      <td>WM</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>CDM</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>3556</th>
      <td>CDM</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>3557</th>
      <td>CB</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>3558</th>
      <td>CB</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>3559</th>
      <td>WB</td>
      <td>WB</td>
    </tr>
    <tr>
      <th>3560</th>
      <td>WM</td>
      <td>WM</td>
    </tr>
    <tr>
      <th>3561</th>
      <td>WM</td>
      <td>WM</td>
    </tr>
    <tr>
      <th>3562</th>
      <td>ST</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>3563</th>
      <td>WM</td>
      <td>WF</td>
    </tr>
    <tr>
      <th>3564</th>
      <td>CAM</td>
      <td>WM</td>
    </tr>
    <tr>
      <th>3565</th>
      <td>ST</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>3566</th>
      <td>ST</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>3567</th>
      <td>WM</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>3568</th>
      <td>ST</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>3569</th>
      <td>CM</td>
      <td>CAM</td>
    </tr>
    <tr>
      <th>3570</th>
      <td>CM</td>
      <td>CDM</td>
    </tr>
    <tr>
      <th>3571</th>
      <td>ST</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>3572</th>
      <td>ST</td>
      <td>ST</td>
    </tr>
    <tr>
      <th>3573</th>
      <td>CM</td>
      <td>CDM</td>
    </tr>
    <tr>
      <th>3574</th>
      <td>WM</td>
      <td>WF</td>
    </tr>
    <tr>
      <th>3575</th>
      <td>CB</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>3576</th>
      <td>CB</td>
      <td>WB</td>
    </tr>
    <tr>
      <th>3577</th>
      <td>CB</td>
      <td>CB</td>
    </tr>
    <tr>
      <th>3578</th>
      <td>GK</td>
      <td>GK</td>
    </tr>
    <tr>
      <th>3579</th>
      <td>CM</td>
      <td>CM</td>
    </tr>
    <tr>
      <th>3580</th>
      <td>WB</td>
      <td>WB</td>
    </tr>
    <tr>
      <th>3581</th>
      <td>CM</td>
      <td>CM</td>
    </tr>
    <tr>
      <th>3582</th>
      <td>WM</td>
      <td>WF</td>
    </tr>
    <tr>
      <th>3583</th>
      <td>CDM</td>
      <td>CDM</td>
    </tr>
  </tbody>
</table>
<p>3584 rows × 2 columns</p>
</div>



# 결론

* WB, ST, GK, CB의 예측률이 매우 높게 나왔고 미드필더의 세부예측률이 낮게 나왔다.


* WF는 아예 예측을 못한것으로 보아 해당 방법에 문제점이 존재한다고 알 수 있다.


* 전체적인 정확도는 약 76%로 k-NN알고리즘보다 약간 높은 정도로 나왔다. 분류에 필요없는 column을 제거한 것과 svm의 더 높은 예측력이 이유인것 같다.
