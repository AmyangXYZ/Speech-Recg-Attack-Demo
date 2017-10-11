# author : Amyang
from flask import Flask, request, render_template, session
from sklearn.externals import joblib
from pydub import AudioSegment
import librosa
from hmmlearn import hmm
import numpy as np
import os
import random
from flag import Flag

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/srv/flask/speech-recg-attack/uploads/'
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
data_folder = '/srv/flask/speech-recg-attack/train_data/'

# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,
                                         covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

def LogMMSE(x, sr):
    '''Speech Enhancement'''
    Slen = int(np.floor(0.02 * sr)) - 1
    noise_frames = 6

    PERC = 50
    len1 = int(np.floor(Slen * PERC / 100))
    len2 = Slen - len1

    win = np.hanning(Slen)
    win = win * len2 / np.sum(win)
    nFFT = 2 * Slen

    x_old = np.zeros(len1)
    Xk_prev = np.zeros(len1)
    Nframes = int(np.floor(len(x) / len2) - np.floor(Slen / len2))
    xfinal = np.zeros(Nframes * len2)

    noise_mean = np.zeros(nFFT)
    for j in range(0, Slen * noise_frames, Slen):
        noise_mean = noise_mean + np.absolute(np.fft.fft(win * x[j:j + Slen], nFFT, axis=0))
    noise_mu2 = noise_mean / noise_frames ** 2

    aa = 0.98
    mu = 0.98
    eta = 0.15
    ksi_min = 10 ** (-25 / 10)

    for k in range(0, Nframes * len2, len2):
        insign = win * x[k:k + Slen]

        spec = np.fft.fft(insign, nFFT, axis=0)
        sig = np.absolute(spec)
        sig2 = sig ** 2

        gammak = np.minimum(sig2 / noise_mu2, 40)

        if Xk_prev.all() == 0:
            ksi = aa + (1 - aa) * np.maximum(gammak - 1, 0)
        else:
            ksi = aa * Xk_prev / noise_mu2 + (1 - aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(ksi_min, ksi)

        log_sigma_k = gammak * ksi / (1 + ksi) - np.log(1 + ksi)
        vad_decision = np.sum(log_sigma_k) / Slen
        if (vad_decision < eta):
            noise_mu2 = mu * noise_mu2 + (1 - mu) * sig2

        A = ksi / (1 + ksi)
        vk = A * gammak
        ei_vk = 0.5 * scipy.special.expn(1, vk)
        hw = A * np.exp(ei_vk)

        sig = sig * hw
        Xk_prev = sig ** 2
        xi_w = np.fft.ifft(hw * spec, nFFT, axis=0)
        xi_w = np.real(xi_w)

        xfinal[k:k + len2] = x_old + xi_w[0:len1]
        x_old = xi_w[len1:Slen]

    if not np.isnan(xfinal[0]):
        return xfinal
    else:
        return x

def Train():
    hmm_models = []
    # Parse the input directory
    for dirname in os.listdir(data_folder):
        print dirname
        # Get the name of the subfolder
        subfolder = os.path.join(data_folder, dirname)
        if not os.path.isdir(subfolder):
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]

        # Initialize variables
        X = np.array([])
        y_words = []

        # Iterate through the audio files (leaving 1 file for testing in each class)
        for filename in [x for x in os.listdir(subfolder) if not x.endswith('.pkl')]:
            # Read the input file
            filepath = os.path.join(subfolder, filename)
            # Convert to wav
            if not filename.endswith('wav'):
                wav = AudioSegment.from_file(filepath)
                os.remove(filepath)
                filepath = filepath.replace('.','_')+'.wav'
                wav.export(filepath,format='wav')
            audio, sampling_freq = librosa.load(filepath)

            # Extract MFCC features
            mfcc_features = librosa.feature.mfcc(y=audio, sr=sampling_freq, n_mfcc=39).T
            # Append to the variable X
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)

            # Append the label
            y_words.append(label)

        print 'X.shape =', X.shape
        # Train and save HMM model
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        joblib.dump(hmm_trainer, subfolder + "/ModelTrained.pkl")
        hmm_trainer = None

def Recg(mfcc_features):
    # Load Models
    hmm_models = []
    for dirname in os.listdir(data_folder):
        # Get the name of the subfolder
        subfolder = os.path.join(data_folder, dirname)

        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]
        hmm_model = joblib.load(subfolder + "/ModelTrained.pkl")
        hmm_models.append((hmm_model, label))

    # Iterate through all HMM models and pick
    # the one with the highest score
    max_score = None
    scores = []
    output_label = None
    for item in hmm_models:
        hmm_model, label = item
        label = unicode(label,'utf-8')
        score = hmm_model.get_score(mfcc_features.T)
        scores.append((label, score))

        if score > max_score:
            max_score = score
            output_label = label

    return output_label, max_score,sorted(scores)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # upload
        file = request.files['file']
        fruit = session['fruit']
        session.pop('fruit',None)
        print fruit
        if file :
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            # Convert to wav
            if file.filename.split('.')[-1] != 'wav' :
                sound = AudioSegment.from_file(os.path.join(app.config['UPLOAD_FOLDER']+file.filename))
                os.remove(app.config['UPLOAD_FOLDER']+file.filename)
                file.filename = file.filename.replace(file.filename.split('.')[-1],'wav')
                sound.export(app.config['UPLOAD_FOLDER']+file.filename, format="wav")

            # Recg
            audio_name = app.config['UPLOAD_FOLDER'] + file.filename
            y, sr = librosa.load(audio_name)
            MFCC = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39)
            result, score, scores = Recg(MFCC)
            if result == fruit and score>-1000:
                s = 'You got it ! ' + Flag
            else :
                s = 'result error or your score is too low, please make sure your score > -1000'
            fruit = None
            return render_template('index.html', title='Biu',**locals())
    else :
        fruits = ['peach', 'apple', 'pineapple', 'orange', 'kiwi', 'banana', 'lime']
        fruit = random.choice(fruits)
        session['fruit'] = fruit
    return render_template('index.html',title='Biu', **locals())

if __name__ == '__main__':
    Train()
    app.run(debug=True, port=8668)
