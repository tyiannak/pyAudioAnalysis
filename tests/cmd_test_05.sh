data_root=$1
echo $data_root
command="python3 ../pyAudioAnalysis/audioAnalysis.py segmentClassifyFileHMM -i \"${data_root}/pyAudioAnalysis/data/scottish.wav\"  --hmm \"${data_root}/pyAudioAnalysis/data/hmmRadioSM\""
echo $command
eval $command