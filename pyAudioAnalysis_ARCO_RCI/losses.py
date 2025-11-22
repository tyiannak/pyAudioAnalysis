"""
ARCO-RCI Loss Functions
========================

Auxiliary loss functions based on ARCO and RCI for training neural networks
on audio and time-series tasks.

The RCI auxiliary loss encourages coherent periodic structure across
multiple channels or leads by maximizing similarity between their arcograms.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from .torch_arc_rci import TorchARCOSpectrum, compute_arcogram_similarity


class RCIAuxLoss(nn.Module):
    """
    RCI-based auxiliary loss for multi-channel signal reconstruction.

    This loss encourages predicted signals to have similar arcogram patterns
    across channels/leads, promoting coherent periodic structure.

    The loss is computed as:
        L = weight * mean(1 - similarity(arcogram_i, arcogram_j))

    where similarity is either cosine similarity or correlation between
    arcograms of different channels.

    Parameters
    ----------
    weight : float
        Loss weight (typically 0.01-0.1)
    fs : float
        Sampling frequency in Hz
    max_q : int
        Maximum denominator for rational arcs
    base_sigma : float
        Base width for Gaussian integration
    device : str or torch.device
        Device to use
    window_size : int
        Window size for arcogram computation
    similarity_metric : str
        'cosine' or 'correlation'
    compare_with_target : bool
        If True, compare predicted with target arcograms
        If False, compare predicted channels with each other

    Examples
    --------
    >>> loss_fn = RCIAuxLoss(weight=0.05, fs=16000, device='cuda')
    >>> pred = torch.randn(8, 3, 1000).cuda()  # Batch of 8, 3 channels, 1000 samples
    >>> target = torch.randn(8, 3, 1000).cuda()
    >>> loss = loss_fn(pred, target)
    >>> print(loss.item())
    """

    def __init__(
        self,
        weight: float = 0.05,
        fs: float = 16000,
        max_q: int = 30,
        base_sigma: float = 0.5,
        device: str = 'cpu',
        window_size: int = 512,
        similarity_metric: str = 'correlation',
        compare_with_target: bool = True
    ):
        super().__init__()

        self.weight = weight
        self.fs = fs
        self.similarity_metric = similarity_metric
        self.compare_with_target = compare_with_target

        # Create ARCO spectrum computer
        self.arco = TorchARCOSpectrum(
            max_q=max_q,
            base_sigma=base_sigma,
            fs=fs,
            window_size=window_size,
            device=device
        )

    def forward(
        self,
        pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute RCI auxiliary loss.

        Parameters
        ----------
        pred : torch.Tensor or Dict[str, torch.Tensor]
            Predicted signals
            - If Tensor: shape (B, n_channels, T)
            - If Dict: maps lead/channel name -> Tensor of shape (B, T)
        target : Optional[torch.Tensor]
            Target signals, shape (B, n_channels, T)
            Only used if compare_with_target=True

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Handle dict input (e.g., multi-lead predictions)
        if isinstance(pred, dict):
            # Stack all leads into a tensor
            pred_list = [pred[k] for k in sorted(pred.keys())]
            pred_tensor = torch.stack(pred_list, dim=1)  # (B, n_leads, T)
        else:
            pred_tensor = pred

        B, n_channels, T = pred_tensor.shape

        if self.compare_with_target:
            # Compare predicted arcograms with target arcograms
            if target is None:
                raise ValueError("target must be provided when compare_with_target=True")

            if isinstance(target, dict):
                target_list = [target[k] for k in sorted(target.keys())]
                target_tensor = torch.stack(target_list, dim=1)
            else:
                target_tensor = target

            # Compute arcograms for each channel
            loss = 0.0
            count = 0

            for i in range(n_channels):
                # Compute arcograms
                pred_arco = self.arco.compute_batch(pred_tensor[:, i, :])
                target_arco = self.arco.compute_batch(target_tensor[:, i, :])

                # Compute similarity
                sim = compute_arcogram_similarity(
                    pred_arco, target_arco, method=self.similarity_metric
                )

                # Loss is 1 - similarity (we want to maximize similarity)
                loss = loss + torch.mean(1.0 - sim)
                count += 1

            loss = loss / count if count > 0 else torch.tensor(0.0, device=pred_tensor.device)

        else:
            # Compare predicted channels with each other (self-consistency)
            loss = 0.0
            count = 0

            # Compute arcograms for all channels
            arcograms = []
            for i in range(n_channels):
                arco = self.arco.compute_batch(pred_tensor[:, i, :])
                arcograms.append(arco)

            # Compare all pairs of channels
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    sim = compute_arcogram_similarity(
                        arcograms[i], arcograms[j], method=self.similarity_metric
                    )
                    loss = loss + torch.mean(1.0 - sim)
                    count += 1

            loss = loss / count if count > 0 else torch.tensor(0.0, device=pred_tensor.device)

        # Apply weight
        return self.weight * loss


class RCIRegularizationLoss(nn.Module):
    """
    RCI regularization loss that encourages high RCI values.

    This loss penalizes signals with low RCI (weak periodicity),
    encouraging the model to generate more periodic/coherent signals.

    Parameters
    ----------
    weight : float
        Loss weight
    fs : float
        Sampling frequency
    max_q : int
        Maximum denominator for rational arcs
    base_sigma : float
        Base width for Gaussian integration
    device : str or torch.device
        Device to use
    window_size : int
        Window size for arcogram computation
    target_rci : float
        Target RCI value (default 0.8)

    Examples
    --------
    >>> loss_fn = RCIRegularizationLoss(weight=0.1, fs=16000, target_rci=0.7)
    >>> pred = torch.randn(8, 1, 1000)
    >>> loss = loss_fn(pred)
    """

    def __init__(
        self,
        weight: float = 0.1,
        fs: float = 16000,
        max_q: int = 30,
        base_sigma: float = 0.5,
        device: str = 'cpu',
        window_size: int = 512,
        target_rci: float = 0.8
    ):
        super().__init__()

        self.weight = weight
        self.target_rci = target_rci

        self.arco = TorchARCOSpectrum(
            max_q=max_q,
            base_sigma=base_sigma,
            fs=fs,
            window_size=window_size,
            device=device
        )

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute RCI regularization loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted signals, shape (B, n_channels, T) or (B, T)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        if pred.ndim == 2:
            # Add channel dimension
            pred = pred.unsqueeze(1)

        B, n_channels, T = pred.shape

        # Compute RCI for all channels
        total_rci_loss = 0.0

        for i in range(n_channels):
            arcograms = self.arco.compute_batch(pred[:, i, :])
            rci = self.arco.compute_rci_batch(arcograms)

            # Loss is squared difference from target
            rci_loss = torch.mean((rci - self.target_rci)**2)
            total_rci_loss = total_rci_loss + rci_loss

        # Average across channels
        loss = total_rci_loss / n_channels

        return self.weight * loss


class ARCODistanceLoss(nn.Module):
    """
    Direct distance between arcograms.

    Measures L1 or L2 distance between predicted and target arcograms.

    Parameters
    ----------
    weight : float
        Loss weight
    fs : float
        Sampling frequency
    max_q : int
        Maximum denominator for rational arcs
    base_sigma : float
        Base width for Gaussian integration
    device : str or torch.device
        Device to use
    window_size : int
        Window size for arcogram computation
    distance_metric : str
        'l1' or 'l2'

    Examples
    --------
    >>> loss_fn = ARCODistanceLoss(weight=0.01, distance_metric='l1')
    >>> pred = torch.randn(8, 2, 1000)
    >>> target = torch.randn(8, 2, 1000)
    >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        weight: float = 0.01,
        fs: float = 16000,
        max_q: int = 30,
        base_sigma: float = 0.5,
        device: str = 'cpu',
        window_size: int = 512,
        distance_metric: str = 'l1'
    ):
        super().__init__()

        self.weight = weight
        self.distance_metric = distance_metric

        self.arco = TorchARCOSpectrum(
            max_q=max_q,
            base_sigma=base_sigma,
            fs=fs,
            window_size=window_size,
            device=device
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute arcogram distance loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted signals, shape (B, n_channels, T) or (B, T)
        target : torch.Tensor
            Target signals, same shape as pred

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        if pred.ndim == 2:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        B, n_channels, T = pred.shape

        total_loss = 0.0

        for i in range(n_channels):
            pred_arco = self.arco.compute_batch(pred[:, i, :])
            target_arco = self.arco.compute_batch(target[:, i, :])

            if self.distance_metric == 'l1':
                dist = torch.mean(torch.abs(pred_arco - target_arco))
            elif self.distance_metric == 'l2':
                dist = torch.mean((pred_arco - target_arco)**2)
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")

            total_loss = total_loss + dist

        loss = total_loss / n_channels

        return self.weight * loss


class CombinedARCOLoss(nn.Module):
    """
    Combined ARCO-based loss with multiple components.

    Combines RCI auxiliary loss, RCI regularization, and arcogram distance.

    Parameters
    ----------
    rci_aux_weight : float
        Weight for RCI auxiliary loss
    rci_reg_weight : float
        Weight for RCI regularization
    arco_dist_weight : float
        Weight for arcogram distance loss
    fs : float
        Sampling frequency
    max_q : int
        Maximum denominator
    device : str or torch.device
        Device

    Examples
    --------
    >>> loss_fn = CombinedARCOLoss(
    ...     rci_aux_weight=0.05,
    ...     rci_reg_weight=0.02,
    ...     arco_dist_weight=0.01
    ... )
    >>> pred = torch.randn(8, 2, 1000)
    >>> target = torch.randn(8, 2, 1000)
    >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        rci_aux_weight: float = 0.05,
        rci_reg_weight: float = 0.02,
        arco_dist_weight: float = 0.01,
        fs: float = 16000,
        max_q: int = 30,
        base_sigma: float = 0.5,
        device: str = 'cpu',
        window_size: int = 512
    ):
        super().__init__()

        self.rci_aux = None
        self.rci_reg = None
        self.arco_dist = None

        if rci_aux_weight > 0:
            self.rci_aux = RCIAuxLoss(
                weight=rci_aux_weight,
                fs=fs,
                max_q=max_q,
                base_sigma=base_sigma,
                device=device,
                window_size=window_size
            )

        if rci_reg_weight > 0:
            self.rci_reg = RCIRegularizationLoss(
                weight=rci_reg_weight,
                fs=fs,
                max_q=max_q,
                base_sigma=base_sigma,
                device=device,
                window_size=window_size
            )

        if arco_dist_weight > 0:
            self.arco_dist = ARCODistanceLoss(
                weight=arco_dist_weight,
                fs=fs,
                max_q=max_q,
                base_sigma=base_sigma,
                device=device,
                window_size=window_size
            )

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with 'total' and individual loss components
        """
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        if self.rci_aux is not None and target is not None:
            loss = self.rci_aux(pred, target)
            losses['rci_aux'] = loss
            total = total + loss

        if self.rci_reg is not None:
            loss = self.rci_reg(pred)
            losses['rci_reg'] = loss
            total = total + loss

        if self.arco_dist is not None and target is not None:
            loss = self.arco_dist(pred, target)
            losses['arco_dist'] = loss
            total = total + loss

        losses['total'] = total
        return losses
