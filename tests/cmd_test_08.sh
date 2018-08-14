data_root=$1
echo $data_root
command1="python3 ../pyAudioAnalysis/audioAnalysis.py speakerDiarization -i \"${data_root}//pyAudioAnalysis/data/diarizationExample.wav\"  --num 4"
echo $command1
eval $command1