try:
    # First try with known format
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H.%M')
except (ValueError, pd.errors.ParserError):
    print("Known format failed, trying automatic parsing...")
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Drop rows where datetime conversion failed
df.dropna(subset=['DateTime'], inplace=True)

# Set datetime index and isolate relevant column
df.set_index('DateTime', inplace=True)

# Ensure the correct column name is used
if 'Traffic_Volume' in df.columns:
    df = df[['Traffic_Volume']]
else:
    print("Column 'Traffic_Volume' not found. Available columns:", df.columns)

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Sequence creation
X = []
y = []
time_step = 60
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i])
    y.append(scaled_data[i])

X, y = np.array(X),np.array(y)
