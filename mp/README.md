# Traffic Light Control Algorithm

Hệ thống điều khiển đèn tín hiệu giao thông thông minh sử dụng thuật toán Max-Pressure và mô phỏng SUMO (Simulation of Urban MObility).

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu hình](#cấu-hình)
- [Các thuật toán điều khiển](#các-thuật-toán-điều-khiển)
- [Cấu trúc dữ liệu](#cấu-trúc-dữ-liệu)
- [Kết quả](#kết-quả)

## 🎯 Tổng quan

Dự án này triển khai và đánh giá các thuật toán điều khiển đèn tín hiệu giao thông sử dụng SUMO simulator. Mục tiêu chính là tối ưu hóa luồng giao thông tại các nút giao thông có đèn tín hiệu thông qua:

- **Thuật toán Max-Pressure**: Điều chỉnh thời gian đèn xanh dựa trên áp lực giao thông (traffic pressure)
- **Thuật toán Fixed-Time**: Sử dụng chu kỳ đèn cố định (baseline)
- **Tích hợp SUMO TraCI**: Giao tiếp thời gian thực với simulator

## ✨ Tính năng

### Các thuật toán điều khiển
- ✅ **Max-Pressure Controller**: Điều khiển thích ứng dựa trên áp lực giao thông
- ✅ **Fixed-Time Controller**: Điều khiển theo chu kỳ cố định
- ✅ Hỗ trợ nhiều nút giao thông đồng thời
- ✅ Tùy chỉnh tham số cho từng nút giao thông

### Mô phỏng và đánh giá
- 🚦 Tích hợp với SUMO qua TraCI interface
- 📊 Thu thập dữ liệu occupancy theo thời gian thực
- 📈 Hỗ trợ chạy với GUI hoặc headless mode
- ⏱️ Lập lịch sự kiện chính xác theo thời gian mô phỏng

### Cấu hình linh hoạt
- 🔧 File cấu hình JSON cho các tham số hệ thống
- 📁 Hỗ trợ nhiều kịch bản (use cases)
- 🎛️ Command-line arguments cho việc thử nghiệm nhanh

## 📂 Cấu trúc dự án

```
trafficlight-algorithm/
├── main.py                          # Entry point chính
├── requirements.txt                 # Dependencies
├── config/
│   └── config.json                  # Cấu hình hệ thống
├── data/
│   ├── caugiay/                     # Dữ liệu mạng lưới Cầu Giấy, Hà Nội
│   │   ├── Caugiay_v1/              # Phiên bản 1 (16 detectors)
│   │   │   ├── caugiay.sumocfg      # SUMO configuration
│   │   │   ├── caugiay_lang_v2.net.xml  # Network file
│   │   │   ├── caugiay.rou.xml      # Routes file
│   │   │   ├── caugiay.add.xml      # Additional structures
│   │   │   ├── net-info.json        # Network metadata
│   │   │   └── e1_*.xml, e2_*.xml   # Detector definitions
│   │   ├── usecase_1/               # Kịch bản 1
│   │   ├── usecase_2/               # Kịch bản 2
│   │   └── usecase_3/               # Kịch bản 3
│   └── phuquoc/                     # Dữ liệu mạng lưới Phú Quốc
│       └── sumo/
│           ├── PhuQuoc/
│           ├── PhuQuoc_v2/
│           └── PhuQuoc_v3/
├── src/
│   ├── cli/
│   │   └── run_args.py              # Command-line argument parser
│   ├── controller/
│   │   ├── base_controller.py       # Abstract base controller
│   │   ├── __init__.py              # Controller registry
│   │   ├── fixed/
│   │   │   └── fixed_time.py        # Fixed-time controller
│   │   ├── maxpressure/
│   │   │   ├── max_pressure.py      # Max-Pressure controller
│   │   │   └── dto/                 # Data transfer objects
│   │   └── webster/
│   │       └── webster.py           # Webster controller (nếu có)
│   ├── service/
│   │   ├── runner.py                # Main simulation runner
│   │   ├── mp_runner.py             # Multi-process runner (nếu cần)
│   │   └── domain/
│   │       └── collected_data.py    # Data structures
│   ├── sim/
│   │   └── traci_interface.py       # SUMO TraCI wrapper
│   └── utils/                       # Utility functions
├── result/                          # Kết quả mô phỏng
└── tests/                           # Unit tests
```

## 💻 Yêu cầu hệ thống

### Phần mềm
- **Python**: >= 3.9
- **SUMO**: >= 1.12.0 (Simulation of Urban MObility)
  - Download: [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/)
  - Cần thêm SUMO vào PATH environment variable

### Thư viện Python
```
numpy
traci
matplotlib
```

## 🚀 Cài đặt

### 1. Clone repository
```powershell
git clone <repository-url>
cd trafficlight-algorithm
```

### 2. Tạo virtual environment (khuyến nghị)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Cài đặt dependencies
```powershell
pip install -r requirements.txt
```

### 4. Cài đặt SUMO
- Download và cài đặt SUMO từ [trang chính thức](https://www.eclipse.org/sumo/)
- Thêm `<SUMO_HOME>/bin` vào PATH
- Kiểm tra: `sumo --version`

## 📖 Sử dụng

### Cách 1: Chạy với task đã cấu hình (VS Code)
```powershell
# Task này đã được cấu hình sẵn
# Nhấn Ctrl+Shift+B hoặc dùng Command Palette > Tasks: Run Task
# Chọn "Run trafficlight main with Caugiay_v1"
```

### Cách 2: Chạy bằng command line

#### Chạy với cấu hình mặc định
```powershell
python main.py
```

#### Chạy với kịch bản Cầu Giấy v1
```powershell
python main.py `
  --sumocfg .\data\caugiay\Caugiay_v1\caugiay.sumocfg `
  --net-info .\data\caugiay\Caugiay_v1\net-info.json `
  --gui
```

#### Chạy với kịch bản usecase 1
```powershell
python main.py `
  --sumocfg .\data\caugiay\usecase_1\caugiay.sumocfg `
  --net-info .\data\caugiay\usecase_1\net-info.json `
  --no-gui
```

#### Chạy với kịch bản Phú Quốc
```powershell
python main.py `
  --sumocfg .\data\phuquoc\sumo\PhuQuoc_v3\phuquoc.sumocfg `
  --net-info .\data\phuquoc\sumo\PhuQuoc_v3\net-info.json `
  --gui
```

### Các tham số command line

```
python main.py [OPTIONS]

Options:
  --config PATH        Path to config.json (default: config/config.json)
  --sumocfg PATH       Path to .sumocfg file (overrides config)
  --net-info PATH      Path to net-info.json (required)
  --gui                Run SUMO with GUI
  --no-gui             Force headless mode
  --log-level LEVEL    Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

### Xem help
```powershell
python main.py --help
```

## ⚙️ Cấu hình

### File `config/config.json`

```json
{
    "sumo": {
        "runner": "traci",
        "begin": 0,
        "end": 1400,
        "gui": true,
        "step_length": 0.1,
        "sample_interval": 10.0
    },
    "controllers": {
        "fixed_time": {
            "name": "fixed_time",
            "params": {
                "program_id": "0"
            }
        },
        "max_pressure": {
            "name": "max_pressure",
            "params": {
                "sample_interval": 10.0,
                "cycling": "exponential",
                "max_delta_green": 5,
                "default_no_signal": 0
            }
        }
    }
}
```

#### Giải thích tham số

**SUMO Configuration**
- `runner`: Interface với SUMO (`"traci"`)
- `begin`: Thời gian bắt đầu mô phỏng (giây)
- `end`: Thời gian kết thúc mô phỏng (giây)
- `gui`: Hiển thị GUI SUMO (true/false)
- `step_length`: Độ dài mỗi time step (giây)
- `sample_interval`: Khoảng thời gian lấy mẫu dữ liệu (giây)

**Controller Configuration**
- `fixed_time`: Cấu hình cho controller chu kỳ cố định
  - `program_id`: ID của chương trình đèn trong SUMO
- `max_pressure`: Cấu hình cho Max-Pressure controller
  - `sample_interval`: Tần suất lấy mẫu dữ liệu
  - `cycling`: Phương pháp tính toán (`"linear"` hoặc `"exponential"`)
  - `max_delta_green`: Thay đổi tối đa thời gian đèn xanh giữa các chu kỳ
  - `default_no_signal`: Giá trị mặc định khi không có tín hiệu

### File `net-info.json`

File này chứa metadata về mạng lưới giao thông:

```json
{
  "tls": {
    "1": {
      "cycle": 98,
      "controller": "max_pressure",
      "edges": {
        "106042137#3": {
          "sat_flow": 5400,
          "length": 102.97,
          "speed": 27.78,
          "detector": ["e2_13", "e2_14", "e2_15"]
        }
      },
      "movements": {
        "106042137#3": {
          "106042137": 0.8,
          "-163053426": 0.1,
          "163053426#6": 0.1
        }
      },
      "phases": {
        "0": {
          "min-green": 10,
          "max-green": 80,
          "movements": [...]
        }
      }
    }
  }
}
```

**Các trường quan trọng:**
- `cycle`: Chu kỳ đèn (giây)
- `controller`: Loại controller (`"max_pressure"` hoặc `"fixed_time"`)
- `edges`: Thông tin các làn đường
  - `sat_flow`: Lưu lượng bão hòa (xe/giờ)
  - `length`: Chiều dài (m)
  - `speed`: Tốc độ giới hạn (m/s)
  - `detector`: Danh sách các detector
- `movements`: Ma trận chuyển động (turning ratio)
- `phases`: Cấu hình các pha đèn
  - `min-green`: Thời gian đèn xanh tối thiểu (giây)
  - `max-green`: Thời gian đèn xanh tối đa (giây)

## 🎮 Các thuật toán điều khiển

### 1. Max-Pressure Controller

**Nguyên lý hoạt động:**

Max-Pressure là thuật toán điều khiển thích ứng dựa trên "áp lực giao thông" (traffic pressure) tại các nút giao.

#### Các bước tính toán:

1. **Thu thập dữ liệu occupancy** từ các detector
   ```
   occupancy[edge] = mean(detector_readings) / 100
   ```

2. **Tính áp lực chuyển động** (movement pressure)
   ```
   pressure[movement] = (occupancy_in - occupancy_out * ratio) * sat_flow
   ```
   - `occupancy_in`: Mật độ xe ở làn vào
   - `occupancy_out`: Mật độ xe ở làn ra
   - `ratio`: Tỷ lệ xe rẽ
   - `sat_flow`: Lưu lượng bão hòa

3. **Tính áp lực pha đèn** (phase pressure)
   ```
   pressure[phase] = sum(pressure[movement] * ratio for movement in phase)
   ```

4. **Phân bổ thời gian đèn xanh**
   
   **Linear cycling:**
   ```
   greentime[phase] = (pressure[phase] / total_pressure) * total_greentime
   ```
   
   **Exponential cycling:**
   ```
   exp_pressure[phase] = exp(pressure[phase] / mean_pressure)
   greentime[phase] = (exp_pressure[phase] / sum_exp_pressure) * total_greentime
   ```

5. **Ràng buộc thời gian**
   - Đảm bảo `min_green <= greentime <= max_green`
   - Tổng thời gian = chu kỳ - lost time

**Tham số:**
- `sample_interval`: 10 giây (mặc định)
- `cycling`: `"linear"` hoặc `"exponential"`
- `min-green`: Thời gian đèn xanh tối thiểu mỗi pha
- `max-green`: Thời gian đèn xanh tối đa mỗi pha

**Ưu điểm:**
- ✅ Thích ứng với lưu lượng thực tế
- ✅ Giảm thời gian chờ trung bình
- ✅ Cân bằng áp lực giữa các hướng

**Nhược điểm:**
- ❌ Phức tạp hơn fixed-time
- ❌ Cần detector chính xác
- ❌ Có thể không ổn định khi lưu lượng thấp

### 2. Fixed-Time Controller

**Nguyên lý:**
- Sử dụng chu kỳ đèn cố định được định nghĩa sẵn trong SUMO
- Không thay đổi theo điều kiện giao thông
- Dùng làm baseline để so sánh

**Ưu điểm:**
- ✅ Đơn giản, dễ triển khai
- ✅ Ổn định, dễ dự đoán
- ✅ Không cần detector

**Nhược điểm:**
- ❌ Không thích ứng với lưu lượng
- ❌ Lãng phí thời gian đèn xanh khi lưu lượng thấp
- ❌ Gây tắc nghẽn khi lưu lượng cao

## 📊 Cấu trúc dữ liệu

### TraCI Interface (`traci_interface.py`)

Wrapper cho SUMO TraCI API, cung cấp các phương thức:

```python
# Khởi động/dừng simulation
start()
close()
step_to(time)
is_completed()

# Thông tin TLS
list_tls_ids()
get_tls_cycle_time(tls_id)
get_tls_splits(tls_id)
set_tls_splits(tls_id, splits)

# Dữ liệu detector
get_lanearea_occupancy(detector_id)

# Thời gian
get_time()
begin_time()
end_time()
```

### Base Controller

Tất cả controllers kế thừa từ `BaseController`:

```python
class BaseController(ABC):
    def __init__(self, tls_id: str, iface, **params):
        self.tls_id = tls_id
        self.cfg = params
        self.iface = iface
    
    @abstractmethod
    def start(self) -> None:
        """Khởi tạo controller"""
        pass
    
    @abstractmethod
    def action(self, t) -> float:
        """
        Thực hiện action tại thời điểm t
        Returns: Thời gian của action tiếp theo
        """
        pass
```

### Runner (`runner.py`)

Quản lý vòng lặp mô phỏng chính:

1. **Khởi tạo** controllers cho tất cả TLS
2. **Lập lịch** các sự kiện:
   - Sampling: Thu thập dữ liệu theo `sample_interval`
   - Controller action: Cập nhật tín hiệu đèn
3. **Step simulation** đến sự kiện tiếp theo
4. **Thu thập kết quả**

## 📈 Kết quả

Kết quả mô phỏng được lưu trong thư mục `result/`.

### Các metrics đánh giá:

1. **Thời gian chờ trung bình** (Average Waiting Time)
2. **Độ dài hàng đợi** (Queue Length)
3. **Tổng thời gian di chuyển** (Total Travel Time)
4. **Throughput**: Số xe hoàn thành hành trình
5. **Fuel consumption & Emissions**

### Phân tích kết quả:

```python
# Đọc kết quả từ SUMO output files
import xml.etree.ElementTree as ET

tree = ET.parse('result/summary-output.xml')
root = tree.getroot()

for step in root.findall('step'):
    time = step.get('time')
    waiting_time = step.get('waitingTime')
    # ... phân tích thêm
```

## 🔧 Mở rộng

### Thêm controller mới

1. Tạo file trong `src/controller/your_controller/`
2. Kế thừa từ `BaseController`
3. Đăng ký với decorator `@register("your_controller")`

```python
from src.controller.base_controller import BaseController
from src.controller import register

@register("your_controller")
class YourController(BaseController):
    def start(self):
        # Khởi tạo
        pass
    
    def action(self, t):
        # Logic điều khiển
        # ...
        return next_action_time
```

4. Thêm cấu hình vào `config/config.json`:

```json
{
  "controllers": {
    "your_controller": {
      "name": "your_controller",
      "params": {
        "param1": "value1"
      }
    }
  }
}
```

### Thêm use case mới

1. Tạo thư mục trong `data/<location>/<usecase_name>/`
2. Chuẩn bị các file SUMO:
   - `*.sumocfg`: SUMO configuration
   - `*.net.xml`: Network file
   - `*.rou.xml`: Routes file
   - `*.add.xml`: Additional structures (detectors)
3. Tạo file `net-info.json` với cấu trúc tương tự
4. Chạy: `python main.py --sumocfg <path> --net-info <path>`

## 🐛 Troubleshooting

### SUMO không tìm thấy
```
Error: SUMO_HOME environment variable not set
```
**Giải pháp**: Thiết lập biến môi trường SUMO_HOME
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

### TraCI connection error
```
Error: Could not connect to TraCI server
```
**Giải pháp**: 
- Kiểm tra SUMO đã cài đặt đúng
- Đảm bảo file `.sumocfg` tồn tại và hợp lệ
- Thử chạy với `--gui` để debug

### Controller không hoạt động
```
Warning: TLS ID X not found in SUMO
```
**Giải pháp**:
- Kiểm tra TLS ID trong `net-info.json` khớp với network
- Xem log file: `data/.../logfile.txt`

### Import error
```
ModuleNotFoundError: No module named 'traci'
```
**Giải pháp**: Cài đặt lại dependencies
```powershell
pip install -r requirements.txt
```

## 📝 Logging

Hệ thống sử dụng Python logging. Điều chỉnh mức độ log:

```powershell
python main.py --log-level DEBUG
```

Log levels: `DEBUG` < `INFO` < `WARNING` < `ERROR`

Log output bao gồm:
- Khởi tạo controllers
- Sampling events
- Controller actions
- Phases pressure và greentime
- Errors và warnings

## 🤝 Đóng góp

Contributions are welcome! Please:
1. Fork repository
2. Tạo branch mới: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add some feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Tạo Pull Request

## 📄 License

[Thêm license của bạn ở đây]

## 👥 Authors

- Bui Chi Toan

## 📧 Contact

- Email: Toan1610@gmail.com
- GitHub: https://github.com/Toan1610

## 🙏 Acknowledgments

- [SUMO - Simulation of Urban MObility](https://www.eclipse.org/sumo/)
- Max-Pressure algorithm references
- Traffic Engineering principles

---

**Last updated**: November 2025
