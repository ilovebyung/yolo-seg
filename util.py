import datetime
import numpy as np
import sounddevice as sd

def get_date_in_yyyymmdd():
  """Returns the current date in YYYYMMDD format."""
  now = datetime.datetime.now()
  return now.strftime("%Y%m%d")

def format_date(date):
  """Returns the current date in YYYYMMDD format."""
  return date.strftime("%Y%m%d")

def get_time_in_mmddss():
  """Returns the current time in mmddss format."""
  now = datetime.datetime.now()
  return now.strftime("%H%M%S")

def format_time(time):
  """Returns the current time in mmddss format."""
  now = datetime.datetime.now()
  return time.strftime("%H%M%S")
  
def get_filename():
  """Returns the current date in YYYYMMDD_mmddss format."""
  now = datetime.datetime.now()
  filename = now.strftime("%Y%m%d") + '_' +  now.strftime("%H%M%S") + '.png'
  return filename

def sound_alarm(duration):
    # Generate a simple sine wave
    sample_rate = 44100 
    frequency = 440 * duration
    t = np.linspace(0, duration, int(sample_rate * duration))
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Play the sound
    sd.play(waveform, sample_rate)
    sd.wait()

if __name__ == "__main__":
    print(get_date_in_yyyymmdd())
    print(get_time_in_mmddss())  
    print(get_filename())
    filename = get_filename()  

    # alarm example usage:
    sound_alarm(1)   