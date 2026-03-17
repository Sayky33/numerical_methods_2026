import requests # pyright: ignore[reportMissingModuleSource]
import numpy as np
import matplotlib.pyplot as plt

def haversine(lat1, long1, lat2, long2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(long2 - long1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def cubic_spline_natural(x, y):
    n = len(x)
    h = np.diff(x)
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)
    B[0] = B[n - 1] = 1
    for i in range(1, n - 1):
        A[i] = h[i - 1]
        B[i] = 2 * (h[i - 1] + h[i])
        C[i] = h[i]
        D[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    for i in range(1, n):
        m = A[i] / B[i - 1]
        B[i] -= m*C[i-1]
        D[i] -= m*D[i-1]

    M =np.zeros(n)
    M[-1] = D[-1]/B[-1]

    for i in range(n-2, -1, -1):
        M[i] = (D[i] - C[i]*M[i+1])/B[i]

    a = y[:-1] 
    b = np.zeros(n-1) 
    c = M[:-1]/2 
    d = np.zeros(n-1)

    for i in range(n-1):
        b[i] = (y[i+1]-y[i])/h[i] - h[i]*(2*M[i]+M[i+1])/6
        d[i] = (M[i+1]-M[i])/(6*h[i])

    
    return a, b, c, d, x

def spline_eval(xi, a, b, c, d, x_nodes):
 
    for i in range(len(x_nodes)-1):
        if x_nodes[i] <= xi <= x_nodes[i+1]:
            dx = xi - x_nodes[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return None

def test_nodes(k):
    indices = np.linspace(0, len(x_full)-1, k, dtype=int)
    x_k = x_full[indices]
    y_k = y_full[indices]
    a_k, b_k, c_k, d_k, x_nodes_k = cubic_spline_natural(x_k, y_k)
    yy_k = np.array([spline_eval(xi, a_k, b_k, c_k, d_k, x_nodes_k) for xi in xx])
    error = np.abs(yy_k - yy_full)
    print(f"\n{k} nodes:")
    print("Max error:", np.max(error))
    print("Average error:", np.mean(error))

    return yy_k, error

locations = [
    {"latitude": 48.164214, "longitude": 24.536044},
    {"latitude": 48.164983, "longitude": 24.534836},
    {"latitude": 48.165605, "longitude": 24.534068},
    {"latitude": 48.166228, "longitude": 24.532915},
    {"latitude": 48.166777, "longitude": 24.531927},
    {"latitude": 48.167326, "longitude": 24.530884},
    {"latitude": 48.167011, "longitude": 24.530061},
    {"latitude": 48.166053, "longitude": 24.528039},
    {"latitude": 48.166655, "longitude": 24.526064},
    {"latitude": 48.166497, "longitude": 24.523574},
    {"latitude": 48.166128, "longitude": 24.520214},
    {"latitude": 48.165416, "longitude": 24.517170},
    {"latitude": 48.164546, "longitude": 24.514640},
    {"latitude": 48.163412, "longitude": 24.512980},
    {"latitude": 48.162331, "longitude": 24.511715},
    {"latitude": 48.162015, "longitude": 24.509462},
    {"latitude": 48.162147, "longitude": 24.506932},
    {"latitude": 48.161751, "longitude": 24.504244},
    {"latitude": 48.161197, "longitude": 24.501793},
    {"latitude": 48.160580, "longitude": 24.500537},
    {"latitude": 48.160250, "longitude": 24.500106},
]

url = "https://api.open-elevation.com/api/v1/lookup"

try:
    response = requests.post(url, json={"locations": locations}, timeout=15)
    response.raise_for_status()
    data = response.json()
    results = data["results"]
except (requests.RequestException, ValueError, KeyError) as e:
    print(e)
    results = [
        {"latitude": 48.164214, "longitude": 24.536044, "elevation": 1264},
        {"latitude": 48.164983, "longitude": 24.534836, "elevation": 1285},
        {"latitude": 48.165605, "longitude": 24.534068, "elevation": 1285},
        {"latitude": 48.166228, "longitude": 24.532915, "elevation": 1333},
        {"latitude": 48.166777, "longitude": 24.531927, "elevation": 1310},
        {"latitude": 48.167326, "longitude": 24.530884, "elevation": 1318},
        {"latitude": 48.167011, "longitude": 24.530061, "elevation": 1318},
        {"latitude": 48.166053, "longitude": 24.528039, "elevation": 1339},
        {"latitude": 48.166655, "longitude": 24.526064, "elevation": 1375},
        {"latitude": 48.166497, "longitude": 24.523574, "elevation": 1417},
        {"latitude": 48.166128, "longitude": 24.520214, "elevation": 1486},
        {"latitude": 48.165416, "longitude": 24.517170, "elevation": 1524},
        {"latitude": 48.164546, "longitude": 24.514640, "elevation": 1553},
        {"latitude": 48.163412, "longitude": 24.512980, "elevation": 1630},
        {"latitude": 48.162331, "longitude": 24.511715, "elevation": 1757},
        {"latitude": 48.162015, "longitude": 24.509462, "elevation": 1794},
        {"latitude": 48.162147, "longitude": 24.506932, "elevation": 1828},
        {"latitude": 48.161751, "longitude": 24.504244, "elevation": 1887},
        {"latitude": 48.161197, "longitude": 24.501793, "elevation": 1975},
        {"latitude": 48.160580, "longitude": 24.500537, "elevation": 1975},
        {"latitude": 48.160250, "longitude": 24.500106, "elevation": 2031},
    ]

n = len(results)
print("Amount of nodes:", n)
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0]

print("\nNode tabulation:")
print("#  | Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | "
    f"{point['longitude']:.6f} | "
    f"{point['elevation']:.2f}")

for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

print("\nTabulation:")
print("# | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

x_full = np.array(distances)
y_full = np.array(elevations)

a_full, b_full, c_full, d_full, x_nodes_full = cubic_spline_natural(x_full, y_full)
xx = np.linspace(x_full[0], x_full[-1], 1000)
yy_full = np.array([spline_eval(xi, a_full, b_full, c_full, d_full, x_nodes_full) for xi in xx])

yy_10, err_10 = test_nodes(10)
yy_15, err_15 = test_nodes(15)
yy_20, err_20 = test_nodes(20)
fg, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(xx, yy_full, label="21 nodes")
ax[0].plot(xx, yy_10, label="10 nodes")
ax[0].plot(xx, yy_15, label="15 nodes")
ax[0].plot(xx, yy_20, label="20 nodes")
ax[0].legend()
ax[0].set_title("Effect of nodes amount")

ax[1].plot(xx, err_10, label="10 nodes")
ax[1].plot(xx, err_15, label="15 nodes")
ax[1].plot(xx, err_20, label="20 nodes")
ax[1].legend()
ax[1].set_title("Approximation error")
plt.show()