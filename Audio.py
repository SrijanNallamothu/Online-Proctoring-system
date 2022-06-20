import speech_recognition as sr
import pyaudio
import wave
import time
import threading
import os

def Audio() :

 def read_audio(stream, filename):
    
     chunk = 1024  
     sample_format = pyaudio.paInt16  
     channels = 2
     fs = 44100  
     seconds = 10 
     filename = filename
     frames = []  
    
     for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Saving the recorded -  WAV file

     wf = wave.open(filename, 'wb')
     wf.setnchannels(channels)
     wf.setsampwidth(p.get_sample_size(sample_format))
     wf.setframerate(fs)
     wf.writeframes(b''.join(frames))
     wf.close()
    
     stream.stop_stream()
     stream.close()

 def convert(i):
     if i >= 0:
         sound = 'record' + str(i) +'.wav'
         r = sr.Recognizer()
        
         with sr.AudioFile(sound) as source:
             r.adjust_for_ambient_noise(source)
             print("Converting Audio To Text and saving to file..... ") 
             audio = r.listen(source)
         try:
             value = r.recognize_google(audio)
             os.remove(sound)
             if str is bytes: 
                 result = u"{}".format(value).encode("utf-8")
             else: 
                 result = "{}".format(value)
 
             with open("audio.txt","a") as f:
                 f.write(result)
                 f.write(" ")
                 f.close()
                
         except sr.UnknownValueError:
             print("")
         except sr.RequestError as e:
             print("{0}".format(e))
         except KeyboardInterrupt:
             pass

 p = pyaudio.PyAudio()  # Create an interface to PortAudio
 chunk = 1024  # Record in chunks of 1024 samples
 sample_format = pyaudio.paInt16  # 16 bits per sample
 channels = 2
 fs = 44100

 def save_audios(i):
     stream = p.open(format=sample_format,channels=channels,rate=fs,frames_per_buffer=chunk,input=True)
     filename = 'record'+str(i)+'.wav'
     read_audio(stream, filename)

 for i in range(30//10): # Number of total seconds to record/ Number of seconds per recording
     t1 = threading.Thread(target=save_audios, args=[i]) 
     x = i-1
     t2 = threading.Thread(target=convert, args=[x]) # send one earlier than being recorded
     t1.start() 
     t2.start() 
     t1.join() 
     t2.join() 
     if i==2:
        flag = True
 if flag:
     convert(i)
     p.terminate()

 import nltk
 from nltk.corpus import stopwords 
 from nltk.tokenize import word_tokenize 
 nltk.download('punkt')

 file = open("audio.txt")
 data = file.read()
 file.close()
 stop_words = set(stopwords.words('english'))   
 word_tokens = word_tokenize(data) 
 filtered_sentence = [w for w in word_tokens if not w in stop_words]  
 filtered_sentence = [] 
  
 for w in word_tokens:  
     if w not in stop_words: 
         filtered_sentence.append(w) 


 f=open('audio_final.txt','w')
 for ele in filtered_sentence:
     f.write(ele+' ')
 f.close()
    

 file = open("Question_paper.txt") # Question paper
 data = file.read()
 file.close()
 stop_words = set(stopwords.words('english'))   
 word_tokens = word_tokenize(data) 
 filtered_questions = [w for w in word_tokens if not w in stop_words]  
 filtered_questions = [] 
  
 for w in word_tokens:   
     if w not in stop_words: 
         filtered_questions.append(w) 
        
 def common_member(a, b):     
     a_set = set(a) 
     b_set = set(b) 
      
     if len(a_set.intersection(b_set)) > 0: 
         return(a_set.intersection(b_set))   
     else: 
         return([]) 

 comm = common_member(filtered_questions, filtered_sentence)
 print('if commom elements :', len(comm))
 print(comm)

 if(len(comm) > 0) :
    
     p = open("audio_log.txt", "a")
     p.write("Audio")
     p.close()


