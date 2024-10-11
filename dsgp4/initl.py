import numpy
import torch

from . import util

def initl(
       xke, j2,
       ecco,   epoch,  inclo,   no,
       opsmode,
       method='n',
       ):

     x2o3   = torch.tensor(2.0 / 3.0);

     eccsq  = ecco * ecco;
     omeosq = 1.0 - eccsq;
     rteosq = omeosq.sqrt();
     cosio  = inclo.cos();
     cosio2 = cosio * cosio;

     ak    = torch.pow(xke / no, x2o3);
     d1    = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq);
     del_  = d1 / (ak * ak);
     adel  = ak * (1.0 - del_ * del_ - del_ *
             (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0));
     del_  = d1/(adel * adel);
     no    = no / (1.0 + del_);

     ao    = torch.pow(xke / no, x2o3);
     sinio = inclo.sin();
     po    = ao * omeosq;
     con42 = 1.0 - 5.0 * cosio2;
     con41 = -con42-cosio2-cosio2;
     ainv  = 1.0 / ao;
     posq  = po * po;
     rp    = ao * (1.0 - ecco);

     if opsmode == 'a':
         #  gst time
         ts70  = epoch - 7305.0;
         ds70 = torch.floor_divide(ts70 + 1.0e-8,1);
         tfrac = ts70 - ds70;
         #  find greenwich location at epoch
         c1    = torch.tensor(1.72027916940703639e-2);
         thgr70= torch.tensor(1.7321343856509374);
         fk5r  = torch.tensor(5.07551419432269442e-15);
         c1p2p = c1 + (2*numpy.pi);
         gsto  = (thgr70 + c1*ds70 + c1p2p*tfrac + ts70*ts70*fk5r) % (2*numpy.pi)
         if gsto < 0.0:
             gsto = gsto + (2*numpy.pi);

     else:
        gsto = util.gstime(epoch + 2433281.5);

     return (
       no,
       method,
       ainv,  ao,    con41,  con42, cosio,
       cosio2,eccsq, omeosq, posq,
       rp,    rteosq,sinio , gsto,
       )
