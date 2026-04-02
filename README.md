## Indoor Positioning using Wi-Fi

### Initial Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### To Scan Wi-Fi: adjust parameters in run.py accordingly and toggle Python Location Services access ON
```bash
source venv/bin/activate
python3 src/data_collection/runs_scans.py -l <LOCATION> -f <ORIENTATION>
```
### Running the application: spin up backend and frontend
## If Python does not have access to location service, it will prompt the user to grant it access
```bash
source venv/bin/activate
python3 run.py --models-dir models/ --port <AVAILABLE_PORT>
```


Rooms scanned
- (S)7.01
- (S)7.02
- (S)7.03
- (S)7.04
- (S)7.05
- (S)7.06
- (S)Hallway

