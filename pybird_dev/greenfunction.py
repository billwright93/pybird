import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1
from scipy.integrate import odeint


class GreenFunction(object):

    def __init__(self, Omega0_m, w=None, quintessence=False, MG=False, Omega_rc=None, nDGP=False):
        #print(quintessence, MG, nDGP)
        self.Omega0_m = Omega0_m
        self.OmegaL_by_Omega_m = (1.-self.Omega0_m)/self.Omega0_m
        self.wcdm = False
        self.quintessence = False
        self.MG = False
        self.nDGP = False
        #print('Om_rc:', Omega_rc)
        if w is not None:
            self.w = w
            if quintessence: self.quintessence = True
            else: self.wcdm = True
        if Omega_rc is not None: #should be elif since don't want both wcDM and nDGP?
            #print('Omega_rc specified, turning on nDGP if you have not done so already!')
            self.Omega_rc = Omega_rc
            if MG:
                if nDGP: print('Omega_rc specified, nDGP and MG selected -- good job!')
                else: print('Omega_rc specified and MG selected, but nDGP not selected -- uh oh!\nTurning on nDGP automatically.')
            else:
                if nDGP: print('Omega_rc specified and nDGP selected, but MG not selected -- uh oh!\nTurning on MG automatically.')
                else: print('Omega_rc specified, but neither nDGP and MG selected -- not good!\nTurning on both MG and nDGP automatically.')
            self.nDGP = True
            self.MG = True #needed?
        else:
            if MG:
                print('nDGP GF:', nDGP)
                if nDGP: print('nDGP and MG selected, but Omega_rc not specified -- not good!')
                else: print('MG selected, but nDGP not selected and Omega_rc not specified -- currently nDGP is the only MG model in PyBird!')
            else:
                if nDGP: print('nDGP selected, but MG not selected and Omega_rc not specified -- uh oh!')
                else: print('No modified gravity stuff specified or selected -- that is fine!')
        '''elif Omega_rc is None and nDGP==True:
            print('nDGP selected but Omega_rc not specified!\nTurning off MG and nDGP.')
            self.MG = False
            self.nDGP = False
        elif Omega_rc is not None and MG==True and nDGP==False:
            print('nDGP selected but Omega_rc not specified!')'''

        self.epsrel = 1e-4

    def C(self, a):
        if self.quintessence: return 1. + (1.+self.w) * self.OmegaL_by_Omega_m * a**(-3.*self.w)
        else: return 1.

    def H(self, a):
        """Conformal Hubble"""
        if self.wcdm or self.quintessence: return ( self.Omega0_m/a + (1.-self.Omega0_m)*a**2 * a**(-3.*(1.+self.w)) )**.5
        else: return (self.Omega0_m/a + (1.-self.Omega0_m)*a**2)**.5

    def H3(self, a):
        return self.C(a)/self.H(a)**3

    def Omega_m(self, a):
        return self.Omega0_m / (self.H(a)**2 * a)

    def H_NC(self, a):
        """Non-Conformal Hubble, H0=1"""
        #if self.nDGP: return np.sqrt(Omega0_m/a/a/a + ((1.-Omega0_m)+ 2.*Omega_rc*(np.sqrt((Omega0_m/a/a/a)/Omega_rc + 1.) - 1.)) + Omega_rc) - np.sqrt(Omega_rc) #nDGP+DE
        #else: return (Omega0_m/a/a/a + (1.-Omega0_m))**0.5 #LCDM
        return (self.Omega0_m/a/a/a + (1.-self.Omega0_m))**0.5 #expansion fixed to LCDM

    def dHda_NC(self, a):
        """Derivative of non-Conformal Hubble w.r.t. a, H0=1"""
        #if self.nDGP: return (-3.*a**(-1. - 3.*(1 + w))*(1 + w)*(1.-Omega0_m) - 3*Omega0_m*a**(-4))/(2.*np.sqrt((1.-Omega0_m)*a**(-3*(1 + w)) + Omega0_m*a**(-3) + Omega_rc)) #nDGP+DE
        #else: return 0.5*(-3.*Omega0_m/a/a/a/a)*(Omega0_m/a/a/a + (1.-Omega0_m))**(-0.5) #LCDM
        return 0.5*(-3.*self.Omega0_m/a/a/a/a)*(self.Omega0_m/a/a/a + (1.-self.Omega0_m))**(-0.5) #expansion fixed to LCDM

    def beta(self, a):
        if self.nDGP: return 1. + self.H_NC(a)*(1.+a*self.dHda_NC(a)/3./self.H_NC(a))/np.sqrt(self.Omega_rc)
        else: return 1.

    def mu(self, a):
        if self.nDGP: return 1.+1./(3.*self.beta(a))
        else: return 1.

    def mu2(self, a):
        if self.nDGP: return -0.5*self.H_NC(a)**2.*(1./(3.*self.beta(a)))**3./self.Omega_rc
        else: return 0.

    def mu22(self, a):
        if self.nDGP: return 0.5*self.H_NC(a)**4.*(1./(3.*self.beta(a)))**5./self.Omega_rc/self.Omega_rc
        else: return 0.

    #######################################################
    ###### Functions to compute MG growth internally ######
    # How to make these self.func form with odeint(func)? #
    #######################################################

    def nDGP_H_NC(a, Omega0_m):
        """Non-Conformal Hubble, H0=1"""
        #if self.MG: return np.sqrt(Omega0_m/a/a/a + ((1.-Omega0_m)+ 2.*Omega_rc*(np.sqrt((Omega0_m/a/a/a)/Omega_rc + 1.) - 1.)) + Omega_rc) - np.sqrt(Omega_rc) #nDGP+DE
        #else: return (Omega0_m/a/a/a + (1.-Omega0_m))**0.5 #LCDM
        return (Omega0_m/a/a/a + (1.-Omega0_m))**0.5 #expansion fixed to LCDM

    def nDGP_dHda_NC(a, Omega0_m):
        """Derivative of non-Conformal Hubble w.r.t. a, H0=1"""
        #if self.MG: return (-3.*a**(-1. - 3.*(1 + w))*(1 + w)*(1.-Omega0_m) - 3*Omega0_m*a**(-4))/(2.*np.sqrt((1.-Omega0_m)*a**(-3*(1 + w)) + Omega0_m*a**(-3) + Omega_rc)) #nDGP+DE
        #else: return 0.5*(-3.*Omega0_m/a/a/a/a)*(Omega0_m/a/a/a + (1.-Omega0_m))**(-0.5) #LCDM
        return 0.5*(-3.*Omega0_m/a/a/a/a)*(Omega0_m/a/a/a + (1.-Omega0_m))**(-0.5) #expansion fixed to LCDM

    def nDGP_beta(a, Omega0_m, Omega_rc):
        """beta function for nDGP""" # Omega_rc as described in 1606.02520 for example
        return 1. + nDGP_H_NC(a, Omega0_m)*(1.+a*nDGP_dHda_NC(a, Omega0_m)/3./nDGP_H_NC(a, Omega0_m))/np.sqrt(Omega_rc)

    def mu_MG(a, Omega0_m, Omega_rc):
        #return 1. #GR
        return 1.+1./(3.*nDGP_beta(a, Omega0_m, Omega_rc)) #nDGP
        #var1 = (k0/a)**2.
        #return 1. + var1/(3.*(var1+pow3(omega0/pow3(a)-4.*(omega0-1.))/(2.*p1/pow2(h0)*pow2(4.-3.*omega0)))) #f(R) Hu- Sawicki

    # how to make this use self?
    def compute_primes_MG(Y, a, Omega0_m, Omega_rc):
        """Second order differential equation for growth factor D extended from Eq.(A.1) of 2005.04805"""
        D, DD = Y
        DDD = (-a*DD - (2.+a*dHda_NC(a, Omega0_m)/H_NC(a, Omega0_m))*a*DD + 1.5*mu_MG(a, Omega0_m, Omega_rc)*Omega_m(a, Omega0_m)*D)/a/a
        return [DD, DDD]

    def D_DD_MG_num(self, a):
        """Solve for growth mode"""
        a_ini = 1e-2
        D_ini_growth = a_ini
        DD_ini_growth = 1.
        a_points = 100
        a_arr = np.linspace(a_ini, a, a_points)
        Y_ini_growth = [D_ini_growth, DD_ini_growth]
        ans = odeint(compute_primes_MG, Y_ini_growth, a_arr, args=(self.Omega0_m, self.Omega_rc))
        D_growth = ans[-1,0]
        DD_growth = ans[-1,1]
        #print(self.Omega_rc)
        return D_growth, DD_growth

    def D_DD_minus_MG_num(self, a):
        """Solve for decay mode"""
        a_ini = 1e-2
        #print(Omega_m_LCDM(a_ini, Omega0_m))
        D_ini_decay = a_ini**(-1.5)
        DD_ini_decay = -1.5*a_ini**(-2.5)
        a_points = 100
        a_arr = np.linspace(a_ini, a, a_points)
        Y_ini_decay = [D_ini_decay, DD_ini_decay]
        ans = odeint(compute_primes_MG, Y_ini_decay, a_arr, args=(self.Omega0_m, self.Omega_rc))
        D_decay = ans[-1,0]
        DD_decay = ans[-1,1]
        return D_decay, DD_decay

    ####################################
    # Back to general PyBird equations #
    ####################################


    def D(self, a):
        """Growth factor"""
        if self.MG: return self.D_DD_MG_num(a)[0]
        elif self.wcdm: return a*hyp2f1((self.w-1)/(2*self.w),-1/(3*self.w),1-(5/(6*self.w)),-(a**(-3*self.w))*self.OmegaL_by_Omega_m)
        else:
            I = quad(self.H3, 0, a, epsrel=self.epsrel)[0]
            return 5 * self.Omega0_m * I * self.H(a) / (2.*a)

    def DD(self, a):
        """Derivative of growth factor"""
        if self.MG: return self.D_DD_MG_num(a)[1]
        elif self.wcdm: return -(a**(-3.*self.w))*self.OmegaL_by_Omega_m*((3*(self.w-1))/(6.*self.w-5.))*hyp2f1(1.5-0.5*(1/self.w),1-(1/(3.*self.w)),2-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)+hyp2f1((self.w-1)/(2.*self.w),-1/(3.*self.w),1-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)
        else: return (2.5-(1.5*self.D(a)/a)) * self.Omega_m(a) * self.C(a)

    def fplus(self, a):
        """Growth rate"""
        return a * self.DD(a) / self.D(a)

    def Dminus(self, a):
        """Decay factor"""
        if self.MG: return self.D_DD_minus_MG_num(a)[0] #D_DD_minus_MG_num(a, self.Omega0_m, self.Omega_rc)[0]
        elif self.wcdm: return a**(-3/2.)*hyp2f1(1/(2.*self.w),(1/2.)+(1/(3.*self.w)),1+(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)
        else: return self.H(a) / (a*self.Omega0_m**.5)

    def DDminus(self, a):
        """Derivative of decay factor"""
        if self.MG: return self.D_DD_minus_MG_num(a)[1] #D_DD_minus_MG_num(a, self.Omega0_m, self.Omega_rc)[1]
        elif self.wcdm: return ((-1+3.*self.w)*hyp2f1(0.5+1/(3.*self.w),1/(2.*self.w),1+5/(6.*self.w),-(a**(-3.*self.w))*(self.OmegaL_by_Omega_m))-(2+3.*self.w)*hyp2f1(1.5+1/(3.*self.w),1/(2.*self.w),1+5/(6.*self.w),-(a**(-3.*self.w))*(self.OmegaL_by_Omega_m)))/(2*(a**(5/2.)))
        else: return -1.5 * self.Omega_m(a) * self.Dminus(a) / a * self.C(a)

    def fminus(self, a):
        """Decay rate"""
        return a * self.DDminus(a) / self.Dminus(a)

    def W(self, a):
        """Wronskian"""
        return self.DDminus(a) * self.D(a) - self.DD(a) * self.Dminus(a)

    #greens functions
    def G1d(self, a, ai):
        return(self.DDminus(ai)*self.D(a)-self.DD(ai)*self.Dminus(a))/(ai*self.W(ai))
    def G2d(self, a, ai):
        return self.fplus(ai)*(self.Dminus(a)*self.D(ai)-self.D(a)*self.Dminus(ai))/(ai*ai*self.W(ai))
    def G1t(self, a, ai):
        return a*(self.DDminus(ai)*self.DD(a)-self.DD(ai)*self.DDminus(a))/(self.fplus(a)*ai*self.W(ai))
    def G2t(self, a, ai):
        return a*self.fplus(ai)*(self.DDminus(a)*self.D(ai)-self.DD(a)*self.Dminus(ai))/(self.fplus(a)*ai*ai*self.W(ai))

    # second order coefficients
    def I1d(self, ai, a):
        if self.MG: return (self.G1d(a,ai)*self.fplus(ai) + G2d(a,ai)*self.mu2(ai)*(1.5*self.Omega_m(ai))**2/self.fplus(ai))*self.D(ai)**2/self.D(a)**2 / self.C(a)
        else: return self.fplus(ai)*self.D(ai)**2*self.G1d(a,ai)/self.D(a)**2 / self.C(a)
    def I2d(self, ai, a):
        if self.MG: return self.G2d(a,ai)*(self.fplus(ai) - self.mu2(ai)*(1.5*self.Omega_m(ai))**2/self.fplus(ai))*self.D(ai)**2/self.D(a)**2 / self.C(a)
        else: return self.fplus(ai)*self.D(ai)**2*self.G2d(a,ai)/self.D(a)**2 / self.C(a)
    def I1t(self, ai, a):
        if self.MG: return (self.G1t(a,ai)*self.fplus(ai) + G2t(a,ai)*self.mu2(ai)*(1.5*self.Omega_m(ai))**2/self.fplus(ai))*self.D(ai)**2/self.D(a)**2 / self.C(a)
        else: return self.fplus(ai)*self.D(ai)**2*self.G1t(a,ai)/self.D(a)**2 / self.C(a)
    def I2t(self, ai, a):
        if self.MG: return self.G2t(a,ai)*(self.fplus(ai) - self.mu2(ai)*(1.5*self.Omega_m(ai))**2/self.fplus(ai))*self.D(ai)**2/self.D(a)**2 / self.C(a)
        else: return self.fplus(ai)*self.D(ai)**2*self.G2t(a,ai)/self.D(a)**2 / self.C(a)

    # second order time integrals
    def mG1d(self, a):
        return quad(self.I1d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2d(self, a):
        return quad(self.I2d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mG1t(self, a):
        return quad(self.I1t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2t(self, a):
        return quad(self.I2t,0,a,args=(a,), epsrel=self.epsrel)[0]

    # quintessence/MG time function
    def G(self, a):
        return self.mG1d(a) + self.mG2d(a)

    # third order coefficients
    def IU1d(self, ai, a):
        if self.MG: return (self.G1d(a,ai)*self.fplus(ai)*self.mG1d(ai) + self.G2d(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IU2d(self, ai, a):
        if self.MG: return ( self.G1d(a,ai)*self.fplus(ai)*self.mG2d(ai) + self.G2d(a,ai)*(1.5*self.Omega_m(ai))**2*( self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai) )/self.fplus(ai) )*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IU1t(self, ai, a):
        if self.MG: return (self.G1t(a,ai)*self.fplus(ai)*self.mG1d(ai) + self.G2t(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IU2t(self, ai, a):
        if self.MG: return ( self.G1t(a,ai)*self.fplus(ai)*self.mG2d(ai) + self.G2t(a,ai)*(1.5*self.Omega_m(ai))**2*( self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai) )/self.fplus(ai) )*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)

    def IV11d(self, ai, a):
        if self.MG: return (self.G1d(a,ai)*self.fplus(ai)*self.mG1t(ai) + self.G2d(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV12d(self, ai, a):
        if self.MG: return self.G2d(a,ai)*(self.fplus(ai)*self.mG1t(ai) - (1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV21d(self, ai, a):
        if self.MG: return (self.G1d(a,ai)*self.fplus(ai)*self.mG2t(ai) + self.G2d(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV22d(self, ai, a):
        if self.MG: return self.G2d(a,ai)*(self.fplus(ai)*self.mG2t(ai) - (1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)

    def IV11t(self, ai,a):
        if self.MG: return (self.G1t(a,ai)*self.fplus(ai)*self.mG1t(ai) + self.G2t(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV12t(self, ai,a):
        if self.MG: return self.G2t(a,ai)*(self.fplus(ai)*self.mG1t(ai) - (1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV21t(self, ai,a):
        if self.MG: return (self.G1t(a,ai)*self.fplus(ai)*self.mG2t(ai) + self.G2t(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV22t(self, ai,a):
        if self.MG: return self.G2t(a,ai)*(self.fplus(ai)*self.mG2t(ai) - (1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)

    # third order time integrals
    def mU1d(self, a):
        return quad(self.IU1d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2d(self, a):
        return quad(self.IU2d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mU1t(self, a):
        return quad(self.IU1t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2t(self, a):
        return quad(self.IU2t,0,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11d(self, a):
        return quad(self.IV11d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12d(self, a):
        return quad(self.IV12d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21d(self, a):
        return quad(self.IV21d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22d(self, a):
        return quad(self.IV22d,0,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11t(self, a):
        return quad(self.IV11t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12t(self, a):
        return quad(self.IV12t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21t(self, a):
        return quad(self.IV21t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22t(self, a):
        return quad(self.IV22t,0,a,args=(a,), epsrel=self.epsrel)[0]

    def Y(self, a):
        return -3/14. + self.mV11d(a) + self.mV12d(a)
