import snntorch as snn

import torch
import torch.nn as nn


class Learner():

        def __init__(self, num_input, num_hidden, num_output, hidden_weights, output_weights, feedback_weights, skip_weights, window, w_inc_hid, w_inc_out, w_inc_fdb, w_inc_skp, w_s_max_hid, w_s_max_out, w_s_max_fdb, w_s_max_skp, device):

                self.device = device
                self.window = window

                self.hidden_weights = hidden_weights
                self.output_weights = output_weights
                self.feedback_weights = feedback_weights
                # self.skip_weights = skip_weights

                # Eligibility traces for synapses: 1 if eligible, 0 otherwise
                self.elig_hidden = torch.zeros((num_hidden, num_input)).to(device)
                self.elig_output = torch.zeros((num_output, num_hidden)).to(device)
                self.elig_feedback = torch.zeros((num_hidden, num_hidden)).to(device)
                # self.elig_skip = torch.zeros((num_output, num_input)).to(device)

                # eligibility timers
                self.last_in = torch.zeros(num_input).to(device)
                self.last_hid = torch.zeros(num_hidden).to(device)
                self.last_out = torch.zeros(num_output).to(device)
                self.last_hid_old = torch.zeros(num_hidden).to(device)

                self.num_input = num_input
                self.num_hidden = num_hidden
                self.num_output = num_output


                # weight_modifiers
                self.weight_mod_hidden = torch.ones_like(self.elig_hidden).to(device)
                self.weight_mod_feedback = torch.ones_like(self.elig_feedback).to(device)
                self.weight_mod_output = torch.ones_like(self.elig_output).to(device)
                # self.weight_mod_skip = torch.ones_like(self.elig_skip).to(device)


                self.w_s_max_hid = w_s_max_hid
                self.w_s_max_out = w_s_max_out
                self.w_s_max_fdb = w_s_max_fdb
                # self.w_s_max_skp = w_s_max_skp
                self.w_inc_hid = w_inc_hid
                self.w_inc_out = w_inc_out
                self.w_inc_fdb = w_inc_fdb
                # self.w_inc_skp = w_inc_skp

                self.fresh = True
                

        # Calculates eligibility traces
        def update(self, in_spk, hid_spk, hid_spk_old, out_spk, do_feedback):

                
                
                if not self.fresh:

                        decay_in = (self.last_in > 0).float().to(self.device)
                        decay_hid = (self.last_hid > 0).float().to(self.device)
                        decay_out = (self.last_out > 0).float().to(self.device)
                        decay_hid_old = (self.last_hid_old > 0).float().to(self.device)

                        self.last_in.sub_(decay_in)
                        self.last_hid.sub_(decay_hid)
                        self.last_out.sub_(decay_out)
                        self.last_hid_old.sub_(decay_hid_old)

                self.fresh = False

                        
                
                self.last_in.add_(torch.mul(in_spk, self.window))
                self.last_hid.add_(torch.mul(hid_spk, self.window))
                self.last_out.add_(torch.mul(out_spk, self.window))
                self.last_hid_old.add_(torch.mul(hid_spk_old, self.window))


                reg_in = torch.zeros(0).to(self.device)
                reg_hid = torch.zeros(0).to(self.device)
                reg_out = torch.zeros(0).to(self.device)
                reg_hid_old = torch.zeros(0).to(self.device)


                torch.remainder(self.last_in, self.window, out=reg_in)
                torch.remainder(self.last_hid, self.window, out=reg_hid)
                torch.remainder(self.last_out, self.window, out=reg_out)
                torch.remainder(self.last_hid_old, self.window, out=reg_hid_old)

                self.last_in.sub_(reg_in)
                self.last_hid.sub_(reg_hid)
                self.last_out.sub_(reg_out)
                self.last_hid_old.sub_(reg_hid_old)

                self.elig_hidden = torch.sub(torch.reshape(self.last_in, (1, self.num_input)), torch.reshape(self.last_hid, (self.num_hidden, 1))) * self.window
                self.elig_output = torch.sub(torch.reshape(self.last_hid, (1, self.num_hidden)), torch.reshape(self.last_out, (self.num_output, 1))) * self.window
                # self.elig_skip = torch.sub(torch.reshape(self.last_in, (1, self.num_input)), torch.reshape(self.last_out, (self.num_output, 1))) * self.window
                

                self.elig_hidden = (self.elig_hidden < 0).float()
                self.elig_output = (self.elig_output < 0).float()
                # self.elig_skip = (self.elig_skip < 0).float()
                

                if do_feedback:
                        self.elig_feedback = torch.sub(torch.reshape(self.last_hid_old, (1, self.num_hidden)), torch.reshape(self.last_hid, (self.num_hidden, 1))) * self.window
                        self.elig_feedback = (self.elig_feedback < 0).float()

        
        def weight_change(self, criticism):


                # Otherwise, do the STDP stuff
                # If the critic returns a negative, we've performed badly.
                if criticism < 0:
                        
                        delta_w_s_hidden = torch.mul(self.weight_mod_hidden, criticism*self.w_inc_hid / self.w_s_max_hid)

                        delta_w_s_output = torch.mul(self.weight_mod_output, criticism*self.w_inc_out / self.w_s_max_out)

                        delta_w_s_feedback = torch.mul(self.weight_mod_feedback, criticism*self.w_inc_fdb / self.w_s_max_fdb)

                        # delta_w_s_skip = torch.mul(self.weight_mod_skip, criticism*self.w_inc_skp / self.w_s_max_skp)

                
                elif criticism > 0:
                        inverse = torch.sub(1, self.weight_mod_hidden / self.w_s_max_hid)
                        delta_w_s_hidden = torch.mul(inverse, criticism*self.w_inc_hid)

                        inverse = torch.sub(1, self.weight_mod_output / self.w_s_max_out)
                        delta_w_s_output = torch.mul(inverse, criticism*self.w_inc_out)

                        inverse = torch.sub(1, self.weight_mod_feedback / self.w_s_max_fdb)
                        delta_w_s_feedback = torch.mul(inverse, criticism*self.w_inc_fdb)

                        # inverse = torch.sub(1, self.weight_mod_skip / self.w_s_max_skp)
                        # delta_w_s_skip = torch.mul(inverse, criticism*self.w_inc_skp)


                else:
                        # criticism = 0: do nothing.
                        return torch.mul(self.hidden_weights, self.weight_mod_hidden), torch.mul(self.output_weights, self.weight_mod_output), torch.mul(self.feedback_weights, self.weight_mod_feedback)
                
                        
                self.weight_mod_hidden.add_(delta_w_s_hidden.mul(self.elig_hidden))
                self.weight_mod_output.add_(delta_w_s_output.mul(self.elig_output))
                self.weight_mod_feedback.add_(delta_w_s_feedback.mul(self.elig_feedback))
                # self.weight_mod_skip.add_(delta_w_s_skip.mul(self.elig_skip))


                # synapses should be multiplied by these values
                return torch.mul(self.hidden_weights, self.weight_mod_hidden), torch.mul(self.output_weights, self.weight_mod_output) , torch.mul(self.feedback_weights, self.weight_mod_feedback)# , torch.mul(self.skip_weights, self.weight_mod_skip)
