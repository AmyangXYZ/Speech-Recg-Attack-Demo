# author: Amyang
import librosa
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
import scipy
import os

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

def invlogamplitude(S):
    """librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

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

def mfcc2wav(input_file, output_file):
    '''inverse MFCC to WAV file'''
    y, sr = librosa.load(input_file)

    # calculate mfcc
    Y = librosa.stft(y)
    mfccs = librosa.feature.mfcc(y)

    # Build reconstruction mappings,
    n_mfcc = mfccs.shape[0]
    n_mel = 128
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 2048
    mel_basis = librosa.filters.mel(sr, n_fft)

    # Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))

    # Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfccs)))

    # Impose reconstructed magnitude on white noise STFT.
    excitation = np.random.randn(y.shape[0])
    E = librosa.stft(excitation)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
    try:
        recon = LogMMSE(recon,sr)
    except:
        pass
    # Listen to the reconstruction.
    librosa.output.write_wav(output_file, recon, sr)

def recg(input_file):
    '''Recognition'''
    # Load Models
    hmm_models = []
    for dirname in os.listdir('train_data/'):
        # Get the name of the subfolder
        subfolder = os.path.join('train_data', dirname)
        if not os.path.isdir(subfolder):
            continue
        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]
        hmm_model = joblib.load(subfolder + "/ModelTrained.pkl")
        hmm_models.append((hmm_model, label))

    # Read input file
    audio, sampling_freq = librosa.load(input_file)

    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sampling_freq, n_mfcc=39).T

    # Define variables
    max_score = None
    output_label = None

    # Iterate through all HMM models and pick
    # the one with the highest score
    for item in hmm_models:
        hmm_model, label = item
        score = hmm_model.get_score(mfcc_features)

        if score > max_score:
            max_score = score
            output_label = label

    # Print the output

    return output_label, max_score

if __name__ == '__main__':
    for i in range(0,100):
        # input_file0 = 'test_data/apple/apple13.wav'
        input_file = 'output/{}.wav'.format(i)
        output_file = 'output/{}.wav'.format(i+1)
        mfcc2wav(input_file, output_file)
        result, score = recg(output_file)
        print "[*] {} Predicted: {}, score:{}".format(i, result, score)

