## 👋 hello

This project is being created based on the project examples below.

https://www.youtube.com/watch?v=4Q3ut7vqD5o&t=364s

https://github.com/roboflow/supervision



## 💻 install

- clone repository and navigate to example directory

    ```bash
    git clone https://github.com/roboflow/supervision.git
    cd supervision/examples/traffic_analysis
    ```

- setup python environment and activate it [optional]

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

- install required dependencies

    ```bash
    pip install -r requirements.txt
    ```

- download `traffic_analysis.pt` and `traffic_analysis.mov` files

    ```bash
    ./setup.sh
    ```

## ⚙️ run

```bash
python script.py \
--source_weights_path data/traffic_analysis.pt \
--source_video_path data/traffic_analysis.mov \
--confidence_threshold 0.3 \
--iou_threshold 0.5 \
--target_video_path data/traffic_analysis_result.mov
```


## vscdoe launch.json
```bash
{
	"version": "0.2.0",
	"configurations": [
		{
			"name": "Python: Current File",
			"type": "python",
			"request": "launch",
			"program": "${file}", // "${workspaceFolder}/test.py",
			"console": "integratedTerminal",
			"justMyCode": true,
			"cwd": "${fileDirname}",
			"args": [
				// "--source_weights_path","data/traffic_analysis.pt",
				"--source_weights_path","data/best.pt",
				
				// "--source_video_path","data/traffic_analysis.mov",
				"--source_video_path","data/9s.mp4",
				
				"--confidence_threshold","0.1",
				"--iou_threshold","0.1",

				// "--target_video_path","data/9s_result.mp4",
				// "--target_video_path","data/traffic_analysis_result.mov",
			], 
		}
	]
}
```

## Polygon Zone

https://roboflow.github.io/polygonzone/


## PyTorch

https://pytorch.org/get-started/locally/