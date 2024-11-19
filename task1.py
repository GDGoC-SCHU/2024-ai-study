import matplotlib.pyplot as plt
import pandas as pd

data = {
    "GDP per capita (USD)": [30000, 40000, 50000, 60000],
    "Life satisfaction": [5.5, 6.0, 6.5, 7.0]
}
df = pd.DataFrame(data)

df.plot(kind="scatter", x="GDP per capita (USD)", y="Life satisfaction")
plt.title("GDP와 삶의 만족도 관계")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.show()