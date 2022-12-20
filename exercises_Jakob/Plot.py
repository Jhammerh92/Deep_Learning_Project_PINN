import deepxde as dde
import numpy as np
# import torch
import matplotlib.pyplot as plt
# import pandas as pd


class Plot:
    def __init__(self, model, colors = None):
        self.model = model
        self.values_to_plot = self.model.disease_model.initial_conditions_keys
        self.sird_values_to_plot = self.model.disease_model.sird_groups
        
        if colors is None:
            self.colors = ['C0','C1','C2','C3']
    
    def _set_color(self, line, colors):
        ''' Set color for each curve in a figure '''
        for idx, color in enumerate(colors):
            line[idx].set_color(color)
    
    def show_deepxde_plot(self):
        dde.saveplot(self.model.losshistory, self.model.train_state, issave=False, isplot=True)
    
    def show_known_and_prediction(self, figsize=(15,8)):
        
        # fig, axes = plt.subplots(1,2
        #                  #, sharex=True
        #                  , sharey=True
        #                  , figsize=figsize)
        fig = plt.figure(figsize=(15,8))
        gs = fig.add_gridspec(2, 2 )
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
        ax3 = fig.add_subplot(gs[:,-1] )
        # ax4 = fig.add_subplot(gs[1,-1], sharex=ax3)
        
        self.plot_known_data([ax1, ax2])
        
        # self.plot_future_prediction([ax3, ax4])
        self.plot_future_prediction(ax3)
    
    def _plot_known_synthetic(self, ax):
        s = 1
        # if 'S' in self.values_to_plot:
        #     line = ax.scatter(self.model.t, self.model.wsol[:,0], color=self.colors[0], s=s)
        # if 'I' in self.values_to_plot:
        #     line = ax.scatter(self.model.t, self.model.wsol[:,1], color=self.colors[1], s=s)
        # if 'R' in self.values_to_plot:
        #     line = ax.scatter(self.model.t, self.model.wsol[:,2], color=self.colors[2], s=s)
        # if 'D' in self.values_to_plot:
        #     line = ax.scatter(self.model.t, self.model.wsol[:,3], color=self.colors[3], s=s)

        for i, label in enumerate(self.sird_values_to_plot):
            if "Im" in label:
                continue
            try:
                line, = ax.plot(self.model.t, self.model.wsol[:,i], color=f"C{i}", ls='-',marker='',markersize=1, lw=1.8, alpha=0.5)
            except:
                print(f"No synthetic data in {label} to plot")
        line.set_label('Synthetic')
        # self._set_color(line, self.colors)
    
    def _plot_known_nn(self, ax, linestyle='--'):
        # if 'S' in self.values_to_plot:
        #     line = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,0], linestyle=linestyle, color=self.colors[0])
        # if 'I' in self.values_to_plot:
        #     line = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,1], linestyle=linestyle, color=self.colors[1])
        # if 'R' in self.values_to_plot:
        #     line = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,2], linestyle=linestyle, color=self.colors[2])
        # if 'D' in self.values_to_plot:
        #     line = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,3], linestyle=linestyle, color=self.colors[3])

        for i, label in enumerate(self.sird_values_to_plot):
            # if label in ['S', 'D']:
            #     continue
            line_s, = ax.plot(self.model.t_nn_best, self.model.wsol_sird_nn_best[:,i], linestyle='--', lw = 1.5, color=f"C{i}")#, label=f"{label} Super Prediction")
        # line_s, = ax.plot(self.model.t_nn_best, self.model.wsol_sird_nn_best[:,i], linestyle='--', lw = 1, color=f"k")#, label=f"{label} Super Prediction")

        for i, label in enumerate(self.values_to_plot):
            line, = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,i], linestyle='-.', lw = 1, color=f"C{i}")#, label=f"{label} Prediction")
        # line, = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,i], linestyle=':', lw = 0.8, color='k', label=f"{label} Prediction")

        line_s.set_label('PINN super prediction')
        line.set_label('PINN prediction')
        # self._set_color(line, self.colors)
    
    def plot_known_data(self, ax):
        ax[0].set_title('Known data')
        if isinstance(ax, list):
            # fig, ax = plt.subplots(2,1)
            # self._plot_known_synthetic(ax[0])
            for i, label in enumerate(self.sird_values_to_plot):
                if "Im" in label:
                    continue
                try:
                    line_synth, = ax[0].plot(self.model.t, self.model.wsol[:,i], color=f"C{i}", ls='-',marker='',markersize=1, lw=1.8, alpha=0.5, label=label)
                except:
                    print(f"No synthetic data in {label} to plot")
            line_synth, = ax[0].plot(self.model.t, self.model.wsol[:,i], color=f"C{i}", ls='-',marker='',markersize=1, lw=1.8, alpha=0.5, label=label)

            for i, label in enumerate(self.sird_values_to_plot):
                line_s, = ax[0].plot(self.model.t_nn_best, self.model.wsol_sird_nn_best[:,i], linestyle='--', lw = 1.5, color=f"C{i}", alpha=1)#, label=f"{label} Super Prediction")
            
            # self._plot_known_nn(ax[1])
            for i, label in enumerate(self.values_to_plot):
                line, = ax[1].plot(self.model.t_nn_best, self.model.wsol_nn_best[:,i], linestyle='--', lw = 1, color=f"C{i}")#, label=f"{label} Prediction")

            self._plot_pred_synthetic(ax[1])

            # ax[0].set_title('Future prediction')
            line_synth.set_label('Synthetic')
            line_s.set_label('PINN super prediction')
            line.set_label('PINN prediction')
            plt.tight_layout()
            ax[0].legend()
            ax[1].legend()
        else:
            self._plot_known_synthetic(ax)
            
            self._plot_known_nn(ax)
            
            # ax.legend()
    
    def _plot_pred_synthetic(self, ax):

        for i, label in enumerate(self.values_to_plot):
            if "Im" in label:
                mul = self.model.init_num_people
            else:
                mul = 1
            try:
                line = ax.plot(self.model.t_synth, self.model.wsol_synth[:,i] * mul, linestyle='-', color=f"C{i}", lw=1.8, alpha=0.5, label=label)
            except:
                pass
                # print(f"No synth Data for label {label}")
        # line[0].set_label('Synthetic')

    
    def _plot_pred_nn(self, ax, linestyle='--'):

        for i, label in enumerate(self.values_to_plot):
            if "Im" in label:
                mul = self.model.init_num_people
            else:
                mul = 1
            line, = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth[:,i] * mul , linestyle=linestyle, color=f"C{i}")
        line.set_label('PINN Parameter PDE prediction')

    
    def plot_future_prediction(self, ax=None):
        if isinstance(ax, list):
            # fig, ax = plt.subplots(2,1)
            self._plot_pred_synthetic(ax[0])
            
            self._plot_pred_nn(ax[1], linestyle='-')
            ax[0].set_title('Future prediction')
            plt.tight_layout()
            ax[0].legend()
        else:
            ax.set_title('Future prediction')
        
            self._plot_pred_synthetic(ax)
            
            self._plot_pred_nn(ax, linestyle='--')
        
            ax.legend()

    def plot_param_history(self):

        if self.model.use_ln_space:
            param_history = np.exp(np.array(self.model.param_history))
        else:
            param_history = np.array(self.model.param_history)
        best_step = self.model.train_state.best_step

        large_axis_plot = False
        nrows = 1

        legend = []
        large_legend = []
        index = []
        large_index = []
        for i, ( key, val) in enumerate(self.model.variables_dict.items()):
            if any(param_history[:,i] > 3.0):
                nrows = 2
                large_axis_plot = True
                large_legend.append(key)
                large_index.append(i)
                continue
            index.append(i)
            legend.append(key)

        ymax = np.amax(param_history[:,index]) * 1.1
        y_delta = ymax - np.amax(param_history[:,index])
        ymin = np.amin(param_history[:,index]) - y_delta
        # ymin_sign = ymin / abs(ymin)
        # ymin = ymin * (1 - ymin_sign * 0.1)


        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(nrows, 1, 1)
        ax.plot(np.arange(len(param_history)), param_history[:,index])
        ax.set_ylim([ymin, ymax])
        ax.legend(legend)
        ax.vlines(best_step, ymin, ymax, 'k', ls='--')

        if large_axis_plot:
            ymax_large = np.amax(param_history[:,large_index]) * 1.1
            ymin_large = np.amin(param_history[:,large_index]) * 0.9
            ax_large = fig.add_subplot(nrows,1, 2)
            ax_large.plot(np.arange(len(param_history)), param_history[:,large_index])
            ax_large.legend(large_legend)
            ax_large .vlines(best_step, ymin_large, ymax_large, 'k', ls='--')

        plt.tight_layout()

    def plot_loss_history(self):

        NUM_COLORS = len(self.model.losshistory.loss_train[0])
        best_step = self.model.train_state.best_step

        color_map = plt.get_cmap('jet')

        nrows = 1
        if self.model.with_softadapt == True:
            nrows = 2

        fig = plt.figure(figsize=(15, 8*nrows))

        ax = fig.add_subplot(nrows, 1, 1)
        ax.set_title("Actual loss values")
        ax.set_prop_cycle(color=[color_map(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        ax.plot(self.model.losshistory.steps, np.linalg.norm(self.model.losshistory.loss_train, axis=1), 'r', lw=1, label = "Loss Norm")
        ax.plot(self.model.losshistory.steps, np.sum(self.model.losshistory.loss_train, axis=1), 'b', lw=1, label = "Loss Sum")
        ax.plot(self.model.losshistory.steps, self.model.losshistory.loss_train, lw=0.7, alpha=0.5)
        ax.set_yscale('log')
        ax.legend( ["Loss Norm", "Loss Sum"] + self.model.PDE_names + list(self.model.loss_points_dict.keys()), ncol=6)
        ax.vlines(best_step, 0, 1.1, 'k', ls='--', lw=0.7)
        ax.hlines(np.linalg.norm(self.model.losshistory.loss_train, axis=1)[best_step], self.model.losshistory.steps[0], self.model.losshistory.steps[-1], 'k', ls='--', lw=1)
        ax.set_ylim([1e-9, 1.1])
        
        
        if nrows == 2:
            ax2 = fig.add_subplot(nrows, 1, 2)
            ax2.set_title("Reweighted loss values")
            ax2.set_prop_cycle(color=[color_map(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
            ax2.plot(self.model.losshistory.steps, self.model.losshistory.loss_weighted, lw=0.7, alpha=0.5)
            ax2.plot(self.model.losshistory.steps, np.linalg.norm(self.model.losshistory.loss_weighted, axis=1), 'r', lw=1, label = "Loss Norm")
            ax2.set_yscale('log')
            ax2.vlines(best_step, 0, 1.1, 'k', ls='--')
            ax2.set_ylim([1e-9, 1.1])
            # ax2.legend(self.model.PDE_names + list(self.model.loss_points_dict.keys()))

        plt.tight_layout()


if __name__ == "__main__":
    pass