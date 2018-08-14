data_root=$1
echo $data_root
command="python3 ../pyAudioAnalysis/audioAnalysis.py fileSpectrogram -i \"${data_root}/pyAudioAnalysis/data/doremi.wav\""
echo $command
eval $command