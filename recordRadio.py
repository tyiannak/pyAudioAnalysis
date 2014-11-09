import sys, os, gi, time
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

GObject.threads_init()
Gst.init(None)

stations = {}
stations["skai"] = "http://netradio.live24.gr/skai1003"
stations["somafm-sonicuniverse"] = "http://uwstream2.somafm.com:8604"

class StreamRecorder():
    
    def __init__(self):
        self._pipeline = Gst.Pipeline()
        self.uri        = None
        self.filename   = None
        self.samplerate = 44100
        self.listen     = False
        
    def _build_pipeline(self):
        
        if not self.uri:
            raise ValueError('Stream URI is not set')
        
        # 1. Create pipeline's elements
        source        = Gst.ElementFactory.make('uridecodebin', 'input_stream')
        splitter      = Gst.ElementFactory.make('tee', 'splitter')
        recording_bin = self._create_recording_bin() if self.filename else None
        listening_bin = self._create_listening_bin() if self.listen else None
        
        # 2. Set properties
        source.set_property('uri', self.uri)
        
        # 3. Add them to the pipeline
        self._pipeline.add(source)
        self._pipeline.add(splitter)
        if recording_bin: self._pipeline.add(recording_bin)
        if listening_bin: self._pipeline.add(listening_bin)
        
        # 4. Link elements
        # 4.a. uridecodebin has a "sometimes" pad (created after prerolling)
        source.connect('pad-added', lambda src, pad: src.link(splitter))
        
        # 4.b. tee has "request" pads (we have to ask for their creation)
        # 
        # NOTE: Python bindings automatically request pads needed
        #splitter.get_request_pad('src_%u') #--> not needed.
        if recording_bin: splitter.link(recording_bin)
        if listening_bin: splitter.link(listening_bin)
            
    def _create_recording_bin(self):
        recording_bin = Gst.Bin.new('recording_bin')
        queue     = Gst.ElementFactory.make('queue', None)
        resampler = Gst.ElementFactory.make('audioresample', None)
        caps      = Gst.ElementFactory.make('capsfilter', None)
        encoder   = Gst.ElementFactory.make('wavenc', None)
        sink      = Gst.ElementFactory.make('filesink', None)
        
        caps.set_property('caps', Gst.Caps.from_string(
            'audio/x-raw, rate={0}'.format(self.samplerate)))
        sink.set_property('location', self.filename)
        
        recording_bin.add(queue, resampler, caps, encoder, sink)
        
        # Gst.Element.link_many() not implemented...
        queue.link(resampler)
        resampler.link(caps)
        caps.link(encoder)
        encoder.link(sink)
        
        recording_bin.add_pad(Gst.GhostPad.new('recording_sink', queue.get_static_pad('sink')))
        return recording_bin
        
    def _create_listening_bin(self):
        listening_bin = Gst.Bin.new('listening_bin')
        queue = Gst.ElementFactory.make('queue', None)
        sink  = Gst.ElementFactory.make('alsasink', None)
        
        #listening_bin.add(queue, sink)
        listening_bin.add(queue)
        listening_bin.add(sink)
        queue.link(sink)
        
        listening_bin.add_pad(Gst.GhostPad.new('listening_sink', queue.get_static_pad('sink')))
        return listening_bin
        
    def start(self):
        self._build_pipeline()
        self._pipeline.set_state(Gst.State.PLAYING)
            
    def stop(self):
        self._pipeline.set_state(Gst.State.NULL)
        
    def bus(self):
        return self._pipeline.get_bus()
 
def recordStation(stationName, outputName, sleepTime = -1, Listen = False):
#	command = "gst-launch-1.0 uridecodebin uri="+stations["skai"]+" ! tee name=t \ 
#	t. ! queue ! audioresample ! audio/x-raw, rate=16000 ! wavenc ! filesink location=" + outputName
#	command = "gst-launch-1.0 uridecodebin uri=" + stations[stationName]+ " 	! tee name=t \ t. ! queue ! audioresample ! audio/x-raw, rate=16000 ! wavenc ! filesink location="+outputName
#	command = "gst-launch-1.0 souphttpsrc location=" + stations[stationName]+ " 	! tee name=t \ t. ! queue ! audioresample ! audio/x-raw, rate=16000 ! wavenc ! filesink location="+outputName
#	os.system(command)

	r = StreamRecorder()
	r.uri = stations[stationName]
	tempName = outputName.replace(".wav", "_temp.wav")
	r.filename = tempName
	r.samplerate = 16000
	r.listen = Listen
	r.start()

	print r.bus()
	if sleepTime<=0:
		raw_input('Press [Enter] to stop')
	else:
		time.sleep(sleepTime)
	r.stop()
	os.system("avconv -i " + tempName + " -ar 16000 -ac 1 -y " + outputName)

	

def main(argv):
	recordStation(argv[1], argv[2], int(argv[3]), False)
	return 0
	
if __name__ == '__main__':
	main(sys.argv)
