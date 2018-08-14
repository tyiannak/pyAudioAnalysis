data_root=$1
echo $data_root
command1="python3 ../pyAudioAnalysis/audioAnalysis.py thumbnail -i \"${data_root}/musicData/AmyWinehouseBacktoblack.wav\" --size 30"
echo $command1
eval $command1