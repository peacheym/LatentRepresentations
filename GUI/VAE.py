import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon
        z = mean + var*epsilon                          # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        # takes exponential function (log var -> var)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


minmax = {'amp_attack': (0.0, 2.5),
          'amp_decay': (0.0, 2.4966),
          'amp_sustain': (0.0, 1.0),
          'amp_release': (0.0, 2.5),
          'filter_attack': (0.0, 2.5),
          'filter_decay': (0.0, 2.49172),
          'filter_sustain': (0.0, 1.0),
          'filter_release': (0.0, 2.5),
          'filter_resonance': (0.0, 0.97),
          'filter_env_amount': (-16.0, 16.0),
          'filter_cutoff': (-0.5, 1.5),
          'osc2_detune': (-1.0, 1.0),
          'lfo_freq': (0.0, 7.5),
          'osc2_range': (-3.0, 4.0),
          'osc_mix': (-1.0, 1.0),
          'freq_mod_amount': (0.0, 1.25992),
          'filter_mod_amount': (-1.0, 1.0),
          'amp_mod_amount': (-1.0, 1.0),
          'osc_mix_mode': (0.0, 1.0),
          'osc1_pulsewidth': (0.0, 1.0),
          'osc2_pulsewidth': (0.0, 1.0),
          'reverb_roomsize': (0.0, 1.0),
          'reverb_damp': (0.0, 1.0),
          'reverb_wet': (0.0, 1.0),
          'reverb_width': (0.0, 1.0),
          'osc2_pitch': (-12.0, 12.0),
          'filter_kbd_track': (0.0, 1.0)}

# TODO: This can defintely be refactored due to the several repeated entries above. Need to take into account the order, and the number of a categorical possibilities for each categorical variable.
PH_COLS = ['amp_attack', 'amp_decay', 'amp_sustain', 'amp_release',
           'filter_attack', 'filter_decay', 'filter_sustain', 'filter_release',
           'filter_resonance', 'filter_env_amount', 'filter_cutoff', 'osc2_detune',
           'lfo_freq', 'osc2_range', 'osc_mix', 'freq_mod_amount',
           'filter_mod_amount', 'amp_mod_amount', 'osc_mix_mode',
           'osc1_pulsewidth', 'osc2_pulsewidth', 'reverb_roomsize', 'reverb_damp',
           'reverb_wet', 'reverb_width', 'osc2_pitch', 'filter_kbd_track',
           'osc1_waveform_0', 'osc1_waveform_1', 'osc1_waveform_2',
           'osc1_waveform_3', 'osc1_waveform_4', 'osc2_waveform_0',
           'osc2_waveform_1', 'osc2_waveform_2', 'osc2_waveform_3',
           'osc2_waveform_4', 'lfo_waveform_0', 'lfo_waveform_1', 'lfo_waveform_2',
           'lfo_waveform_3', 'lfo_waveform_4', 'lfo_waveform_5', 'lfo_waveform_6',
           'osc2_sync_0', 'osc2_sync_1', 'filter_type_0', 'filter_type_1',
           'filter_type_2', 'filter_type_3', 'filter_type_4', 'filter_slope_0',
           'filter_slope_1', 'freq_mod_osc_0', 'freq_mod_osc_1', 'freq_mod_osc_2']
