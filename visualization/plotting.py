import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import os


CLASSIFICATION_COLORS = [
    "black", "yellow", "lightgreen", "indigo", "orange", "pink", "peru",
    "crimson", "aqua", "dodgerblue", "slategrey", "b", "red", "darkcyan",
    "grey", "olive", "green", "gold",
]

TSNE_COLORS = CLASSIFICATION_COLORS[1:]


def draw_classification_map(pred, label, name, acc, save_dir='./results',
                            scale=4.0, dpi=400):
    """Draw and save the classification map.

    Args:
        pred: Predicted labels (1-indexed, 0 is background)
        label: Ground truth map (2D array, 0 is background)
        name: Dataset name for save path
        acc: Accuracy value for filename
        save_dir: Directory to save the result
        scale: Figure scale factor
        dpi: Figure DPI
    """
    indices = np.where(label != 0)
    label[indices] = pred

    rgb_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(CLASSIFICATION_COLORS):
        rgb = np.array(mcolors.to_rgb(color)) * 255
        rgb_label[label == i] = rgb.astype(np.uint8)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(rgb_label)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.savefig(
        f'{save_dir}/{name}/pred_{acc:.5f}.png',
        format='png', transparent=True, dpi=dpi, pad_inches=0
    )


# def draw_tsne(features, labels, acc, dataname, save_dir='./results', title=None):
#     """Draw and save the t-SNE visualization.
#
#     Args:
#         features: Feature tensor from the model
#         labels: Ground truth labels
#         acc: Accuracy value for filename
#         dataname: Dataset name for save path
#         save_dir: Directory to save the result
#         title: Optional plot title
#     """
#
#     # 1. 全局字体与样式设置 (Times New Roman 是学术论文常用字体)
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体与Times New Roman匹配
#     plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
#
#     X = features.cpu().numpy()
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     X_tsne = tsne.fit_transform(X)
#
#     plt.figure(figsize=(10, 10))
#     unique_labels = np.unique(labels)
#     for label_val in unique_labels:
#         mask = labels == label_val
#         color_idx = int(label_val) % len(TSNE_COLORS)
#         plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], marker='o',
#                     color=TSNE_COLORS[color_idx], s=10)
#
#     if title is not None:
#         plt.title(title)
#
#     plt.rc('font', family='Times New Roman')
#     plt.savefig(
#         f'{save_dir}/{dataname}/tsne_{acc:.5f}.pdf',
#         bbox_inches='tight', pad_inches=0
#     )
#     # plt.show()


def draw_tsne(features, labels, acc, dataname, save_dir='./results', title=None):
    """Draw and save the t-SNE visualization meeting academic standards.

    Args:
        features: Feature tensor from the model
        labels: Ground truth labels
        acc: Accuracy value for filename
        dataname: Dataset name for save path
        save_dir: Directory to save the result
        title: Optional plot title
    """
    # 1. 全局字体与样式设置 (Times New Roman 是学术论文常用字体)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'STIXGeneral']
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体匹配 STIX
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 数据准备
    X = features.cpu().numpy()
    # 归一化 (帮助 t-SNE 收敛)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # 3. t-SNE 降维
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    # 4. 创建图与坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # 5. 绘制散点
    for label_val in unique_labels:
        mask = labels == label_val
        color_idx = int(label_val) % len(TSNE_COLORS)

        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            marker='o',
            color=TSNE_COLORS[color_idx],
            s=25,  # 适当增大点的大小
            alpha=0.8,  # 增加透明度以展现簇的密度
            edgecolors='none',  # 去除点边缘，避免大量点重叠时变黑
            # label=f'Class {int(label_val)}'  # 添加图例标签
        )

    # 6. 坐标轴与网格设置
    # ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='normal')
    # ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='normal')

    # 去除上、右边框 (学术论文常采用简约风格)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # 设置刻度大小
    ax.tick_params(axis='both', which='major', labelsize=12)

    # t-SNE 的坐标轴绝对数值通常无实际物理意义，部分论文选择隐藏刻度数字
    # ax.set_xticks([])
    # ax.set_yticks([])

    # 7. 添加标题
    if title is not None:
        ax.set_title(title, fontsize=16, pad=15)

    # 8. 添加图例 (如果类别数不多于20个才显示图例，否则过于拥挤)
    if num_classes <= 20:
        legend = ax.legend(
            loc='best',
            fontsize=10,
            frameon=False,  # 去除图例边框
            ncol=1  # 单列显示，若类别多可改为 2
        )

    # 9. 保持横纵比例一致，防止簇的形状被拉伸
    ax.set_aspect('equal', adjustable='datalim')

    # 10. 保存图像
    save_path = f'{save_dir}/{dataname}'
    os.makedirs(save_path, exist_ok=True)  # 确保目录存在

    plt.savefig(
        f'{save_path}/tsne_{acc:.5f}.pdf',
        bbox_inches='tight',
        pad_inches=0.1,  # 略微增加边缘留白
        dpi=600  # 设置高 DPI 保证矢量清晰度
    )
    plt.close(fig)  # 释放内存

