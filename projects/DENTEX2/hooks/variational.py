import numpy as np
import torch


def initialise_prior(n_classes, n_volunteers, alpha_diag_prior):
    """
    Create confusion matrix prior for every volunteer - the same prior for each volunteer
    :param n_classes: number of classes (int)
    :param n_volunteers: number of crowd members (int)
    :param alpha_diag_prior: prior for confusion matrices is assuming reasonable crowd members with weak dominance of a
    diagonal elements of confusion matrices, i.e. prior for a confusion matrix is a matrix of all ones where
    alpha_diag_prior is added to diagonal elements (float)
    :return: numpy nd-array of the size (n_classes, n_classes, n_volunteers)
    """
    alpha_volunteer_template = np.ones((n_classes, n_classes), dtype=np.float64) + alpha_diag_prior * np.eye(n_classes)
    cm = np.tile(np.expand_dims(alpha_volunteer_template, axis=2), (1, 1, n_volunteers))

    return cm


def VB_iteration(X, nn_output, alpha_volunteers, alpha0_volunteers):
    """
    performs one iteration of variational inference update for BCCNet (E-step) -- update for approximating posterior of
    true labels and confusion matrices
    N - number of data points
    J - number of true classes
    L - number of classes used by volunteers (normally L == J)
    W - number of volunteers

    :param X: N X W volunteer answers, -1 encodes a missing answer
    :param nn_output: N X J logits (not a softmax output!)
    :param alpha_volunteers: J X L X W - current parameters of posterior Dirichlet for confusion matrices
    :param alpha0_volunteers: J X L -  parameters of the prior Dirichlet for confusion matrix
    :return: q_t - approximating posterior for true labels, alpha_volunteers - updated posterior for confusion matrices
    """
    ori_cms = alpha_volunteers.copy()

    alpha_volunteers = torch.from_numpy(alpha_volunteers).to(nn_output)
    alpha0_volunteers = torch.from_numpy(alpha0_volunteers).to(nn_output)

    ElogPi_volunteer = expected_log_Dirichlet_parameters(alpha_volunteers)

    # q_t
    q_t, Njl = expected_true_labels(X, nn_output, ElogPi_volunteer)

    # q_pi_workers
    alpha_volunteers = alpha0_volunteers + Njl

    return q_t, alpha_volunteers.cpu().numpy().astype(ori_cms.dtype)


def VB_posterior(X, alpha_volunteers):
    alpha_volunteers = torch.from_numpy(alpha_volunteers).to(X.device)

    ElogPi_volunteer = expected_log_Dirichlet_parameters(alpha_volunteers)

    flat_nn_output = torch.zeros((X.shape[0], alpha_volunteers.shape[0])).to(X.device)
    q_t, _ = expected_true_labels(X, flat_nn_output, ElogPi_volunteer)

    return q_t


def expected_log_Dirichlet_parameters(param):
    if param.ndim == 1:
        return torch.digamma(param) - torch.digamma(torch.sum(param))
    else:
        return torch.digamma(param) - torch.digamma(torch.sum(param, dim=1, keepdim=True))


def expected_true_labels(X, nn_output, ElogPi_volunteer):
    N, W = X.shape  # N = Number of subjects, W = Number of volunteers.
    J = ElogPi_volunteer.shape[0]  # J = Number of classes
    L = ElogPi_volunteer.shape[1] # L = Number of classes used by volunteers

    ElogPi_volunteer = torch.cat((
        ElogPi_volunteer,
        torch.zeros_like(ElogPi_volunteer[:, :1])  # when X == -1
    ), dim=1)
    X = X.unsqueeze(1)
    if X.dtype == torch.int64:
        index_arrays = (torch.arange(J).reshape(1, -1, 1), X, torch.arange(W))
        rho = nn_output + torch.sum(ElogPi_volunteer[index_arrays], axis=-1)
    else:
        index_arrays0 = (torch.arange(J).reshape(1, -1, 1), torch.zeros(N, 1, W).long(), torch.arange(W))
        index_arrays1 = (torch.arange(J).reshape(1, -1, 1), torch.ones(N, 1, W).long(), torch.arange(W))
        tmp = (1 - X) * ElogPi_volunteer[index_arrays0] + X * ElogPi_volunteer[index_arrays1]
        rho = nn_output + torch.sum(tmp, axis=-1)

    rho -= torch.amax(rho, 1, keepdim=True)

    q_t = torch.exp(rho) / torch.clamp(torch.sum(torch.exp(rho), 1, keepdim=True), min=1e-60)
    q_t = torch.clamp(q_t, min=1e-60)

    if X.dtype == torch.int64:
        mask = X.reshape(N, 1, 1, W) == torch.arange(L).to(X)[:, None]  # N, 1, L, W
    else:
        mask = torch.stack([1 - X, X], axis=2)
    Njl = torch.sum(q_t.reshape(N, J, 1, 1) * mask, dim=0)

    return q_t, Njl
