import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def preprocess_matrix(mat, use_pca=False, pca_dim=32, pca_model=None, threshold=0.0, use_lda=False, lda_dim=None, lda_model=None, label=None):
    """
    mat: (116, 116) numpy array
    use_pca: bool, whether to apply PCA
    pca_dim: target dimension for PCA
    pca_model: sklearn PCA object (fit on train set, transform on all)
    use_lda: bool, whether to apply LDA
    lda_dim: target dimension for LDA
    lda_model: sklearn LDA object (fit on train set, transform on all)
    label: int or None, required for LDA fitting
    threshold: 임계값, abs(value) < threshold이면 0으로 만듦
    Returns: 1D numpy array (preprocessed feature)
    전처리 순서:
      1. 대각선 0
      2. upper triangle flatten
      3. z-score 정규화
      4. thresholding
      5. (옵션) PCA/LDA
    """
    # 1. upper triangle만 flatten (맨 처음)
    vec = mat[np.triu_indices_from(mat, k=1)]
    # 2. Z-score 정규화
    vec = (vec - vec.mean()) / (vec.std() + 1e-8)
    # 4. thresholding
    if threshold > 0.0:
        vec[np.abs(vec) < threshold] = 0.0
    # 5. (옵션) LDA or PCA
    if use_lda:
        assert lda_dim is not None, 'lda_dim must be specified for LDA.'
        if lda_model is None:
            assert label is not None, 'label must be provided for LDA fitting.'
            lda_model = LinearDiscriminantAnalysis(n_components=lda_dim)
            vec = lda_model.fit_transform(vec.reshape(1, -1), [label]).flatten()
        else:
            vec = lda_model.transform(vec.reshape(1, -1)).flatten()
    elif use_pca:
        if pca_model is None:
            pca_model = PCA(n_components=pca_dim)
            vec = pca_model.fit_transform(vec.reshape(1, -1)).flatten()
        else:
            vec = pca_model.transform(vec.reshape(1, -1)).flatten()
    return vec

