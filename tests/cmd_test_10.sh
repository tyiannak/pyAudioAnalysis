data_root=$1
echo $data_root
command1="python3 ../pyAudioAnalysis/audioAnalysis.py featureVisualization -i \"${data_root}/musicData\""
echo $command1
eval $command1