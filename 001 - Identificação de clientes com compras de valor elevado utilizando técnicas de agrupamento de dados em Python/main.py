from sklearn.cluster import KMeans

# Carregar os dados de transações de compras em um DataFrame do Pandas
# Aqui, vamos assumir que a variável "con" já esteja configurada 
df = pd.read_sql("SELECT * FROM transactions", con)

# Selecionar apenas as colunas de interesse (ID do cliente e valor da compra)
X = df[['customer_id', 'purchase_value']]

# Inicializar o modelo K-Means com 2 clusters
kmeans = KMeans(n_clusters=2)

# Treinar o modelo usando os dados de compras
kmeans.fit(X)

# Prever o cluster para cada registro de compra
predictions = kmeans.predict(X)

# Adicionar a coluna de cluster ao DataFrame
df['cluster'] = predictions

# Filtrar os clientes no cluster de compras de valor elevado
high_value_customers = df[df['cluster'] == 1]

# Exibir a lista de clientes com compras de valor elevado
print(high_value_customers['customer_id'].unique())