#!/usr/bin/python3
import time
import tkinter
from tkinter import ttk
from ttkthemes import ThemedTk
import libmapper as mpr
import torch
import torch.nn as nn
from VAE import Model, Encoder, Decoder, PH_COLS, minmax as mm
import json
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import random

class PyTorchModelHandler:
    
    model = None
    def __init__(self):
        self.minmax = mm
        self.latent_dim = 2
        
    def load_model(self, latent_dim="2D"):
        # self.latent_dim = int(latent_dim.split("D")[0])
        # self.model = torch.load("{}model.pt".format(latent_dim))
        self.model = torch.load("trained_vae.pt")
        self.model.eval()
        
    def infer_from_model(self, value):
        if self.latent_dim == len(value):
            return self.model.Decoder(torch.tensor(value).to('cpu'))
        else:
            print("Latent Vector Shape does not match Latent Dimension")
            
    # This function is used to re-scale the data into the correct format.
    def unnormalize(self, normalized_value, minmax):
        return format(normalized_value * (minmax[1] - minmax[0]) + minmax[0], '.4f')
                
    def build_format_preset(self, value):
        
        data = self.infer_from_model(value)
        
        # Rebuild Continuous Columns via un-normalization.
        cols = ['amp_attack', 'amp_decay', 'amp_sustain', 'amp_release', 'filter_attack', 'filter_decay', 'filter_sustain', 'filter_release', 'filter_resonance', 'filter_env_amount', 'filter_cutoff', 'osc2_detune', 'lfo_freq', 'osc2_range',
                'osc_mix', 'freq_mod_amount', 'filter_mod_amount', 'amp_mod_amount', 'osc_mix_mode', 'osc1_pulsewidth', 'osc2_pulsewidth', 'reverb_roomsize', 'reverb_damp', 'reverb_wet', 'reverb_width', 'osc2_pitch', 'filter_kbd_track']
        preset_dict = {}  # No preset name for live patch gen
        for i in cols:
            preset_dict[i] = self.unnormalize(data[PH_COLS.index(i)].item(), self.minmax[i])

        # Rebuild Categorical columns via arg-max
        preset_dict['osc1_waveform'] = torch.argmax(
            data[PH_COLS.index('osc1_waveform_0'):PH_COLS.index('osc1_waveform_4')]).item()
        preset_dict['osc2_waveform'] = torch.argmax(
            data[PH_COLS.index('osc2_waveform_0'):PH_COLS.index('osc2_waveform_4')]).item()
        preset_dict['lfo_waveform'] = torch.argmax(
            data[PH_COLS.index('lfo_waveform_0'):PH_COLS.index('lfo_waveform_6')]).item()
        preset_dict['osc2_sync'] = torch.argmax(
            data[PH_COLS.index('osc2_sync_0'):PH_COLS.index('osc2_sync_1')]).item()
        preset_dict['filter_type'] = torch.argmax(
            data[PH_COLS.index('filter_type_0'):PH_COLS.index('filter_type_4')]).item()
        preset_dict['filter_slope'] = torch.argmax(
            data[PH_COLS.index('filter_slope_0'):PH_COLS.index('filter_slope_1')]).item()
        preset_dict['freq_mod_osc'] = torch.argmax(
            data[PH_COLS.index('freq_mod_osc_0'):PH_COLS.index('freq_mod_osc_2')]).item()

        # Rebuild non-learned columns (Statically)
        preset_dict['portamento_mode'] = 0
        preset_dict['portamento_time'] = 0
        preset_dict['keyboard_mode'] = 0
        preset_dict['filter_vel_sens'] = 1
        preset_dict['amp_vel_sens'] = 1
        preset_dict['distortion_crunch'] = 0
        preset_dict['master_vol'] = 0.6

        return preset_dict

                
class GuiHandler:
    def __init__(self, pytorchHandler, signals, width=1000, height=1000, min=-4, max=4) -> None:
        
        self.pytorchHandler = pytorchHandler
        self.signals = signals
        
        self.old_time = time.time()
        
        self.width = width
        self.height = height
        self.min = min
        self.max = max
                
        self.landmark_size = 18
        self.landmark_color = "#a3bbb6"
        
        self.saved_locs = None
        self.saved_node = None 
        self.update_signals = True  
                
        self.root = ThemedTk(theme="yaru")
        self.root.title("Latent Space Explorer")
        self.root.configure(background="white")
        self.root.resizable(0, 0)
        self.root.bind("<BackSpace>", self.handle_backspace)
    

                
        # create a menubar
        menubar = tkinter.Menu(self.root)
        self.root.config(menu=menubar)

        # create a menu
        file_menu = tkinter.Menu(menubar)

        # add a menu item to the menu
        file_menu.add_command(
            label='Exit',
            command=self.root.destroy
        )

        # add the File menu to the menubar
        menubar.add_cascade(
            label="File",
            menu=file_menu
        )
            
        self.bg_color = "#ffffff"
        self.paint = tkinter.Canvas(self.root, width=self.width, height=self.height, background=self.bg_color)
        self.paint.config(cursor="none")
        
        self.paint.bind('<Motion>', self.callback)
        self.paint.bind("<Button-1>", self.click)
        self.paint.bind("<Button-3>", self.handle_alt)
        self.paint.pack()
            
        self.latent_vis_data = pd.read_csv("timbral_vis2.csv")
        self.vis_param = "filter_cutoff"
        self.latent_vis_points = []
        self.latent_coords = []
                
        self.init_latent_viz()
        
        param_menu = tkinter.Menu(self.root)
        # param_menu.add_command(
        #     label="none",
        #     command=lambda: self.init_latent_viz()
        # )
        for i in self.latent_vis_data.columns[2:]: # Disregard unnamed, X, Y cols.
            param_menu.add_command(
                label=i,
                command=lambda i = i : self.update_latent_viz(i)
            )
        menubar.add_command(label='Show Param Coordinates', command=lambda: self.draw_latent_space("./latent_coords.csv", 5, "#8884FF"))
        menubar.add_cascade(label="Visualize Timbral Quality", menu=param_menu)
                
        # Draw the navigator lines & crosshair
        self.circle = self.paint.create_oval(0, 0, 0, 0, fill="#e8e8e8", outline="#e8e8e8")
        self.xline = self.paint.create_line(0, 0, 0, 0, fill="#e8e8e8", width=3)
        self.yline = self.paint.create_line(0, 0, 0, 0, fill="#e8e8e8", width=3)
    
    
        # Run main functions
        self.do_poll()
        self.root.mainloop()
    

    
    def play_synth_sound(self):
        if self._state != "PLAY":
            self.old_time = time.time() # Reset timer on first play

            if len(self._curr_sound_clips) > 0:
                self._curr_sound = random.choice(self._curr_sound_clips)
                self._curr_sound_clips.pop(self._curr_sound_clips.index(self._curr_sound))
            else:
                print("out of sounds")
                # TODO: Transition to next group of sounds.
        print('./Sounds/{}/{}'.format(self._curr_sound_group, self._curr_sound))
        playsound('./Sounds/{}/{}'.format(self._curr_sound_group, self._curr_sound), block=False)
        self.update_state("PLAY")
        
    def save_patch(self):
 
        new_time = time.time()
        self.update_state("SAVE")
        
        # TODO: 1) Log the latent vector associated with the saved patch
        
        print(self.saved_loc)
        
        # TODO: 2) Log the elapsed time for this trial.

        print("Elapsed Time:", new_time - self.old_time)

    
    def next_sound(self):
        self.paint.delete(self.saved_node)
        self.update_signals = True
        self.update_state("NEXT")
    
    def do_poll(self):
        dev.poll(20)
        self.root.after(5, self.do_poll)
        
    def update_state(self, state):
        if state == "PLAY": 
            self.save_patch_btn.config(state=tkinter.DISABLED)
            self.next_btn.config(state=tkinter.DISABLED)
            
            self._state = "PLAY"
            return
        
        if state == "READY_TO_SAVE":
            self.save_patch_btn.config(state=tkinter.NORMAL)

        if state == "SAVE":
            self.next_btn.config(state=tkinter.NORMAL)
            self.save_patch_btn.config(state=tkinter.DISABLED)
            self.play_sound_btn.config(state=tkinter.DISABLED)
            
            self._state = "SAVE"
            return
        if state == "NEXT":
            self.save_patch_btn.config(state=tkinter.DISABLED)
            self.next_btn.config(state=tkinter.DISABLED)
            self.play_sound_btn.config(state=tkinter.NORMAL)
                
            self._state = "NEXT"
            return
        
        
    def init_latent_viz(self, size=15):
        for _, row in self.latent_vis_data.iterrows():
            x = self.ls_to_canvas_coords(row['X'])  
            y = self.ls_to_canvas_coords(row['Y'])
            node = self.paint.create_oval(0, 0, 0, 0, fill=self.bg_color, outline=self.bg_color)
            self.paint.coords(node, x-size, y-size, x+size, y+size)
            self.latent_vis_points.append(node)

    def update_latent_viz(self, param):
        # Remove all latent coords, only show one or the other.
        for node in self.latent_coords:
            self.paint.delete(node)
        viridis = mpl.colormaps['plasma']
        print(param)
        for index, row in self.latent_vis_data.iterrows():
            ratio = 100
            if param == "reverb":
                ratio = 1
            color = mpl.colors.to_hex(viridis(row[param] / ratio)) # Extract color from colormap based on value in data
            self.paint.itemconfig(self.latent_vis_points[index], fill=color)
            self.paint.itemconfig(self.latent_vis_points[index], outline=color)

    def callback(self, event):
        if self.update_signals:
            self.draw(event.x, event.y)
            # Update the canvas

            fixed_x = self.canvas_to_ls_coords(event.x)
            fixed_y = self.canvas_to_ls_coords(event.y)
            
            # TODO: Link to the PytorchModelHandler
            value = [fixed_x, fixed_y]
            result = self.pytorchHandler.build_format_preset(value)
            # # Update libmapper signals
            for i in result:
                signals[i].set_value(float(result[i]))
    
    
    def draw_latent_space(self, file_path, size, color):
        
        for node in self.latent_vis_points:
            self.paint.itemconfig(node, fill=self.bg_color)
            self.paint.itemconfig(node, outline=self.bg_color)

        
        try:
            latent_coords = pd.read_csv(file_path)
            for _, row in latent_coords.iterrows():
                x = self.ls_to_canvas_coords(row['X'])
                y = self.ls_to_canvas_coords(row['Y'])
                
                # TODO: Consider storing a reference to each of these nodes.
                node = self.paint.create_oval(0, 0, 0, 0, fill=color, outline=color)
                self.latent_coords.append(node)
                self.paint.coords(node, x-size, y-size, x+size, y+size)
        except Exception as e:
            print(e)

    
    def click(self, event):
        if self._state == "SAVE":
            return
        if self.update_signals:
            self.update_signals = False
            self.update_state("READY_TO_SAVE")
            
        self.saved_loc = (event.x, event.y)
        
        if self.saved_node:
            self.paint.delete(self.saved_node)
        node = self.paint.create_oval(0, 0, 0, 0, fill=self.landmark_color, outline=self.landmark_color)
        self.saved_node = node
        self.draw(event.x, event.y)
        
        self.paint.coords(node,  event.x-self.landmark_size, event.y-self.landmark_size, event.x+self.landmark_size, event.y+self.landmark_size)
        # with open("state.json", "w") as outfile:
        #     json.dump(self.saved_locs, outfile)

    def handle_alt(self, event):
        self.update_signals = not self.update_signals
        
    def draw(self, x, y):
        self.paint.coords(self.circle, x-10, y-10, x+10, y+10)
        self.paint.coords(self.xline, 0, y, self.height, y)
        self.paint.coords(self.yline, x, 0, x, self.width)
        
    def handle_backspace(self, event):
        if self.saved_loc and self._state != "SAVE":
            self.saved_loc = None
            self.paint.delete(self.saved_node)
            self.update_state("PLAY")
            self.update_signals = True
    
    def rescale(self, val, new_min, new_max, old_min, old_max):
        return float((new_max-new_min)*(val-old_min)/(old_max-old_min)+new_min)

    def canvas_to_ls_coords(self, val):
        return self.rescale(val, self.min, self.max, 1, self.height)

    def ls_to_canvas_coords(self, val):
        return self.rescale(val, 1, self.height, self.min, self.max)

if __name__ == "__main__":
    
    pth = PyTorchModelHandler()
    pth.load_model("2D")

    graph = mpr.Graph()
    graph.set_interface("wlp0s20f3")
    dev = mpr.Device("slider", graph)

    signals = {}


    params = ['amp_attack', 'amp_decay', 'amp_sustain', 'amp_release', 'osc1_waveform', 'filter_attack', 'filter_decay', 'filter_sustain', 'filter_release', 'filter_resonance', 'filter_env_amount', 'filter_cutoff', 'osc2_detune', 'osc2_waveform', 'master_vol', 'lfo_freq', 'lfo_waveform', 'osc2_range', 'osc_mix', 'freq_mod_amount', 'filter_mod_amount',
            'amp_mod_amount', 'osc_mix_mode', 'osc1_pulsewidth', 'osc2_pulsewidth', 'reverb_damp', 'reverb_roomsize', 'reverb_wet', 'reverb_width', 'distortion_crunch', 'osc2_sync', 'portamento_time', 'keyboard_mode', 'osc2_pitch', 'filter_type', 'filter_slope', 'freq_mod_osc', 'filter_kbd_track', 'filter_vel_sens', 'amp_vel_sens', 'portamento_mode']

    for i in params:
        sig_out = dev.add_signal(mpr.Direction.OUTGOING, i,
                                1, mpr.Type.FLOAT, None, None, None)
        signals[i] = sig_out
    
    gh = GuiHandler(pth, signals, min=-4, max=4, width=800, height=800)

