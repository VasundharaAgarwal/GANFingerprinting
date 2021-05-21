import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import fp_utilities as fp_util


def plot_prec_recall_chart(fp_type, preds, ax, first_plot = True):
  """Plots bar chart of precision and recall values computed from predictions

  :param fp_type : specifies which of Handcrafted or Learned was used to generate 
                 predictions
  :param preds : predictions to use for generating precision and recall values
  :param ax : Axes object to plot to
  :param first_plot : specifies whether it is the first plot in the figure
  """

  scores = precision_recall_fscore_support(ground_truth, preds, labels = labels)
  precision_scores = scores[0]
  recall_scores = scores[1]
  
  ind = np.arange(4)  
  width = 0.4  
  
  rects1 = ax.bar(ind - width/2, precision_scores, width, label='Precision')
  rects2 = ax.bar(ind + width/2, recall_scores, width, label='Recall')

 
  

  for rect in rects1+rects2:
    height = rect.get_height()
    ax.annotate('{}'.format(round(height, 3)),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",  
                ha='center', va='bottom')

  ax.set_xticks(ind)
  ax.set_xticklabels(labels)
  ax.set_ylim(0,1.2)
  ax.axhline(y=0.25,linestyle='dashed', color='k')
  ax.set_yticks([0.00, 0.20, 0.25, 0.40, 0.60, 0.80, 1.00,1.20]) 
  
  if(first_plot):
    ax.set_yticklabels([0.00, 0.20, 'Baseline' ,0.40,0.60,0.80,1.00])
  else:
    ax.set_yticklabels([]) 
    ax.legend()

  ttl = ax.title
  ttl.set_text(fp_type)
  ttl.set_fontweight('bold')
  ttl.set_position([0.5, -0.18])
  

def plot_fingerprints(fp_extraction_method, denoising_method="median blur"):
  """Visualise the fingerprints extracted using a particular method
  
  :param fp_extraction_method : specifies which of handcrafted (Marra) and learned fingerprints
                                (Yu) must be visualised
  :param denoising_method : specifies denoising method in case of handcrafted 
                            (default is median blur)
  """

  fps = fp_util.load_fingerprints(fp_extraction_method, denoising_method)
  fps_combined = np.array(fps)
  val_min, val_max = np.amin(fps_combined), np.amax(fps_combined)

  fp_labels = ['real', 'GAN_1', 'GAN_2', 'GAN_3']
  fig, ax = plt.subplots(1,4, figsize = (19,13))
              
  for i in range(4):
    im = ax[i].imshow(fps[i].reshape(28,28), vmin = val_min, vmax = val_max)
    ax[i].set_title(fp_labels[i].replace('_', ' '), fontsize=20)
    ax[i].axis('off')
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
  plt.savefig('Fingeprints_{}.pdf'.format(fp_extraction_method), bbox_inches='tight')

def _visualise_data(data, vis_type, ax):
  pca_n_components = 2 if vis_type == 'PCA' else 50
  pca = PCA(n_components= pca_n_components)
  pca_dim_reduced_fps = pca.fit_transform(data)

  if (vis_type == 'PCA'):
    df = pd.DataFrame(pca_dim_reduced_fps, columns = ['dim-1','dim-2'])
  else : 
    tsne_dim_reduced_fps = TSNE(n_components=2, verbose=1).fit_transform(pca_dim_reduced_fps)
    df = pd.DataFrame(tsne_dim_reduced_fps, columns = ['dim-1','dim-2'])
  df['source'] = ground_truth

  sns.scatterplot(ax = ax,
      x="dim-1", y="dim-2",
      hue="source",
      palette=sns.color_palette("hls", 4),
      data=df,
      legend="full",
      alpha=0.3
  )
  ax.set_xlabel('')
  ax.set_ylabel('')
  


def visualise_fp_dist(fp_extraction_method, vis_type, ax):
  """ Visualise dimensionality-reduced image residuals extracted using a particular method

  :param fp_extraction_method : specifies which of Handcrafted and Learned must be
                                used to extract image residuals
  :param vis_type : specifies method to use for dimensionality reduction
  :param ax : Axes object to plot to
  """
  fps_test_imgs = []
  for gan_num in range(0,4):
    test_imgs = fp_util.load_test_images(gan_num)
    for i in range(len(test_imgs)):
      img = test_imgs[i]
      fps_test_imgs.append(fp_util.extract_fingerprint(img, fp_extraction_method).flatten()) 
  
  visualise_data(fps_test_imgs, vis_type, ax)
  
def visualise_test_img_dist(vis_type, ax):
  """ Visualise dimensionality-reduced test images

  :param vis_type : specifies method to use for dimensionality reduction
  :param ax : Axes object to plot to
  """
  test_imgs_flattened = []
  for gan_num in range(0,4):
    test_imgs = fp_util.load_test_images(gan_num)
    for i in range(len(test_imgs)):
      test_imgs_flattened.append(test_imgs[i].flatten())
  visualise_data(test_imgs_flattened, vis_type, ax )
  
def plot_atk_accuracy_lines(atk_strengths, marra_accuracies, yu_accuracies):
  """ Visualise evolution of accuracy of both algorithms with increasing attack strength

  :param atk_strengths : attack strengths to plot on the x-axis
  :param marra_accuracies : accuracy values of Handcrafted to plot on y-axis
  :param yu_accuracies : accuracy values of Learned to plot on y-axis
  """

  plt.rcParams['axes.labelsize'] =  16
  plt.rcParams['legend.fontsize']  =  12
  
  fig, ax = plt.subplots(1, 1, figsize = (7,5))
  
  ax.plot(atk_strengths, marra_accuracies, label="Handcrafted")
  ax.plot(atk_strengths, yu_accuracies, label="Learned")
  ax.set_yticks([0.00, 0.20, 0.25, 0.40, 0.60, 0.80, 1.00]) 
  ax.set_yticklabels([0.00, 0.20, 'Baseline' ,0.40,0.60,0.80,1.00])
  ax.axhline(y=0.25,linestyle='dashed', color='k')
  ax.set_xlabel('Attack Strength')
  ax.set_ylabel('Accuracy of attribution')
  ax.legend()

def plot_atk_images(atk_images, atk_strengths_plot):
  """ Visualise attacked images

  :param atk_images : attacked images to visualise
  :param atk_strengths_plot : attack strengths to label the visualised images with
  """
  fig, ax = plt.subplots(1, 6, figsize=(40,6))
  for i in range(6):
    ax[i].imshow(atk_images[i])
    ax[i].axis('off')

    ttl = ax[i].title
    ttl.set_text(atk_strengths_plot[i])
    ttl.set_fontsize(32)
    ttl.set_fontweight('bold')
    ttl.set_position([0.5, -0.15])

#adapted from official matplotlib implementation
def _plot_cm_custom(cm_display, include_values=True, cmap='Blues',
             xticks_rotation='horizontal', values_format=None,
             ax=None, colorbar=True):
  """ Returns customised confusion matrix,
      documentation as here - https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/metrics/_plot/confusion_matrix.py#L78
  """
  
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = cm_display.confusion_matrix
        n_classes = cm.shape[0]
        cm_display.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1000)
        cm_display.text_ = None
        cmap_min, cmap_max = cm_display.im_.cmap(0), cm_display.im_.cmap(256)

        if include_values:
            cm_display.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], '.2g')
                    if cm.dtype.kind != 'f':
                        text_d = format(cm[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                cm_display.text_[i, j] = ax.text(
                    j, i, text_cm,
                    ha="center", va="center",
                    color=color)

        if cm_display.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = cm_display.display_labels
        if colorbar:
            fig.colorbar(cm_display.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=display_labels,
               yticklabels=display_labels,
               ylabel="True source",
               xlabel="Predicted source")
        ax.xaxis.labelpad = 14
        ax.set_xlabel('Predicted source', fontsize=15)

        ax.yaxis.labelpad = 5
        ax.set_ylabel('True source', fontsize=15)

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        cm_display.figure_ = fig
        cm_display.ax_ = ax
        return cm_display


def plot_confusion_matrix(fp_type, preds, ax):
  """Plots confusion matrices for visualising predictions given as argument

  :param fp_type : specifies which of Handcrafted or Learned was used to generate 
                 predictions
  :param preds : predictions to visualise
  :param ax : Axes object to plot to
  """
  cm = confusion_matrix(ground_truth, preds, labels)
  cm_plot = ConfusionMatrixDisplay(cm, labels)
  plot_cm_custom(cm_plot, ax=ax, values_format='d')
  
  ttl = ax.title
  ttl.set_text(fp_type)
  ttl.set_fontweight('bold')
  ttl.set_position([0.5, -0.3])
    
def plot_corr_histograms(fp_extraction_method, attack_mode='none'):
  """ Visualise distribution of correlation coefficients between residuals of test images from 
      the candidate sources and each of the source fingerprints
  
  :param fp_extraction_method : specifies which of handcrafted (Marra) and 
                              learned (Yu) fingerprints to extract from test images
  :param denoising_method : specifies denoising method in case of handcrafted 
                            (default is median blur)
  """ 
  plt.rcParams['axes.labelsize'] =  16
  plt.rcParams['legend.fontsize']  =  12
  plt.rcParams['xtick.labelsize'] = 14
  plt.rcParams['ytick.labelsize'] = 14   
  fig, ax = plt.subplots(1,4,figsize=(16,5), sharex=True, sharey=True)
  for i in range(4):
    coefs_real, coefs_gan_1, coefs_gan_2, coefs_gan_3 = fp_util.compute_corr_coeff(i, fp_extraction_method, attack_mode)
    ax[i].hist(coefs_real, color='red', alpha= 0.25, label="real")
    ax[i].hist(coefs_gan_1, color='blue', alpha = 0.2, label="GAN 1")
    ax[i].hist(coefs_gan_2, color='yellow', alpha = 0.2, label="GAN 2")
    ax[i].hist(coefs_gan_3, color='green', alpha = 0.2, label="GAN 3")
    ax[i].set_xlabel('Correlation')
  plt.legend()

 