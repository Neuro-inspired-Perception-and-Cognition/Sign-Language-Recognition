# Sign Language Fingerspelling Recognition on SpiNNaker

This repository contains the code accompanying the paper:   
**“Neuromorphic visual attention for Sign-language recognition on SpiNNaker”** by S. Liskova, O. Vedmedenko, M. Fatahi, M. Hoffmann, M. Furlong and G. D'Angelo, 2026  



The codebase provides a complete pipeline for event-based sign language fingerspelling recognition, covering data preprocessing from raw event streams, region-of-interest- selection via sVA attention model, spike encoding, spiking neural network (SNN) training,   
and deployment on neuromorphic hardware.

![Architecture diagram](media/sign_recognition_system_diagram.png)
# Pipeline overview
The repository is organized to reflect the full experimental pipeline used in the paper.  

### Dataset acquisition
##### ASL-DVS dataset
•	The original ASL-DVS dataset is available at: [Google drive](https://drive.google.com/drive/folders/1tK5OY3pkjppYwAnLF8bnxGdaFbYEA8iY).  
•	The exact form of ASL-DVS used in our paper can be found at: [Google drive](https://drive.google.com/drive/folders/1ti0ou-iGFHyBpfTag3ftHcD25nXTtrfj?usp=sharing).  
##### Sign Language MNIST dataset
•	The Sign Language MNIST dataset (original RGB dataset) was converted to events using `rgb_to_events/SL_MNIST_to_DVS_events.py`.  
•	The IEBCS simulator used for RGB-to-event conversion is available at: [github](https://github.com/neuromorphicsystems/IEBCS/tree/main).  
•	The exact form of the converted SL MNIST event dataset is included in: [Google drive](https://drive.google.com/drive/folders/1ti0ou-iGFHyBpfTag3ftHcD25nXTtrfj?usp=sharing).
### Region-of-interest (ROI) selection with the sVA attention mechanism
•	The script `sVA_roi_selection/apply_attention_extract_ROI.py` applies visual attention mechanism onto event-based data in .aedat4 format, reporting the most salient point. Then, a square-shaped region of interest around this point is selected and downsampled to the size of fingerspelling recognition (sR) model input size. Subsequently, the events are transformed into spike tensors ready for injection into the recognition network.

### Training of the proposed spiking architecture
•	Training was performed as set up in the `cpu_training/snn_training.ipynb` notebook.  

### Deployment and inference on SpiNNaker hardware
•	Trained model parameters obtained from the CPU-based training pipeline can be found in `inference_on_spinnaker/model_weights/`. These weights are formatted to match the neuron and synapse constraints of the SpiNNaker platform.  
•	The SpiNNaker inference is run using `inference_on_spinnaker/run_spinnaker_inference.py`, loading the trained weights.  
 
# Repository structure
```
.
├── cpu_training/
│   ├── model_weights/
│   ├── snn_training_helper_functions.py
│   └── snn_training.ipynb
│
├── inference_on_spinnaker/
│   ├── model_weights/
│   ├── run_spinnaker_inference.py
│   └── spinnaker_connect_helpers.py
│
├── simulate_events_from_rgb/
│   ├── events_to_spikes.py
│   └── encoding_utils.py
│
├── rgb_to_events/
│   └── SL_MNIST_to_DVS_events.py
│
├── sVA_roi_selection/
│   └── apply_attention_extract_ROI.py
│
├── requirements.txt
└── README.md
```

