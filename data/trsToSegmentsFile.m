function trsToSegmentsFile(fileName, maxTime)
% example
% trsToSegmentsFile('/home/tyiannak/ResearchData/AUDIO/speakerClusteringData/canal9New/05-09-28_manual.trs')

[pathstr, name, ext] = fileparts(fileName);


addpath('/home/tyiannak/Research/MATLAB_CODE/XML/')
addpath('/home/tyiannak/Research/MATLAB_CODE/speakerDiarization/')
[Times, Labels] = readTrsFileSpeakers(fileName, 0.1);

wavFileName = strrep(fileName, '_manual.trs', '.wav')
[x,fs] = wavread(wavFileName);

I1 = find(Labels(1:end-1)==0 & Labels(2:end)>0)+1
I2 = find(Labels(1:end-1)>0 & Labels(2:end)==0)
Labels = Labels(I1(1):I2(end));
Times = Times(I1(1):I2(end));

Labels = Labels(1:maxTime/0.1);
Times = Times(1:maxTime/0.1);
T1 = I1(1)*0.1; T2 = I2(end)*0.1;
x = x(round(T1*fs):round(T2*fs));
x = x(1:round(maxTime*fs));

uLabels = unique(Labels);
for i = 1:length(Labels)
    Labels(i) = find(uLabels==Labels(i));
end

plot(Times, Labels);




[numOfSegments, segs, classes] = flags2segsANDclasses(Labels, 0.1)
fp = fopen([name '.segments'], 'w');
for i=1:size(segs)                
    fprintf(fp,'%.2f,%.2f,%s\n',segs(i,1), segs(i,2), ['spk'  num2str(classes(i))]);
end
fclose(fp);
wavwrite(x, fs, 16, [name '.wav']);
