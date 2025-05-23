import torch

def rhsburger(t,u,*args):
  visc = args[0]
  twopik = args[1]
  x_hat = torch.fft.rfft(u)
  dudx = torch.fft.irfft(twopik*x_hat)
  dudx_2 =torch.fft.irfft((twopik**2)*x_hat)
  return visc*(dudx_2)-(u*dudx)