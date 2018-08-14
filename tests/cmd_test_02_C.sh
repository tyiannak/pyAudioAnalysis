data_root=$1
echo $data_root
command="python3 ../pyAudioAnalysis/audioAnalysis.py classifyFile -i \"${data_root}/musicalGenreClassification/Blues/bb king - guess who.wav\"  --model svm_rbf --classifier \"${data_root}/pyAudioAnalysis/data/svmMusicGenre6\""
echo $command
eval $command

python3 audioAnalysis.py classifyFile -i music_genre_small/rock/The_Mayan_Factor-Warflower.mp3 --model svm --classifier music_genre_small_model
command="python3 ../pyAudioAnalysis/audioAnalysis.py classifyFile -i \"${data_root}/musicalGenreClassification/Blues/bb king - guess who.wav\"  --model knn --classifier \"${data_root}/pyAudioAnalysis/data/knnMusicGenre6\""
echo $command
eval $command