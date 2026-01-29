# Sign-Language-Recognition

This repository contains the code accompanying the paper:

“[Paper title]”
[Authors], [Conference / Journal], [Year]



The codebase provides a complete pipeline for event-based sign language recognition, covering data preprocessing from raw event streams, spike encoding, spiking neural network (SNN) training, and deployment on neuromorphic hardware.

![Architecture diagram](media/sign_recognition_system_diagram.png)

The repository is organized to reflect the full experimental pipeline used in the paper:
	•	Dataset aquisition: - the original ASL-DVS dataset is available at:  
                            - the exact form of ASL-DVS that was used in our paper can be found at:  
                            - The SL MNIST dataset (original RGB dataset: link) was converted to events using the `simulate_events_from_RGB/SL_MNIST_to_DVS_events.py` for our purposes. The IEBCS simulator can be found at:   https://github.com/neuromorphicsystems/IEBCS/tree/main.
                            - The exact form of the SL MNIST after the conversion can be found at:   

	•	Region-of-interest (ROI) selection with the sVA attention mechanism: The script `sVA_roi_selection/apply_attention_extract_ROI.py` applies visual attention mechanism onto event-based data in .aedat4 format, reportig the most salient point. Then, a square-shaped region of interest around this point is selected and downsampled to the size of fingerspelling recognition (sR) model input size. Subsequently, the events are transformed into spike tensors ready for injection into the recognition network.

	•	Training of the proposed spiking architecture
    - Training was performed as set up in the `cpu_training/snn_training.ipynb` notebook.
	•	Deployment and inference on SpiNNaker hardware



The original ASL-DVS dataset available at: https://drive.google.com/drive/folders/1tK5OY3pkjppYwAnLF8bnxGdaFbYEA8iY  
datasets available at: https://drive.google.com/drive/folders/1ti0ou-iGFHyBpfTag3ftHcD25nXTtrfj?usp=sharing
IEBCS: https://github.com/neuromorphicsystems/IEBCS/tree/main  


