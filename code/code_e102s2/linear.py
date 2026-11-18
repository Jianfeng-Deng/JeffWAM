"""CPU-only standardization and class-balanced L2 logistic regression using existing SciPy."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
class StandardScaler:
 def fit(self,x):
  self.mean_=np.mean(x,axis=0,dtype='f8');v=np.mean((x-self.mean_)**2,axis=0);self.scale_=np.sqrt(v);self.scale_[self.scale_<1e-12]=1;return self
 def transform(self,x):return (x-self.mean_)/self.scale_
class LogisticRegression:
 def __init__(self,C=1.,class_weight='balanced',max_iter=250,**kwargs):self.C=C;self.max_iter=max_iter
 def fit(self,x,y):
  x=np.asarray(x,dtype='f8');y=np.asarray(y,dtype='f8');n,d=x.shape;positive=y.sum();w=np.where(y==1,n/(2*positive),n/(2*(n-positive)));lam=1/(self.C*n)
  def objective(v):
   z=x@v[:-1]+v[-1];loss=np.mean(w*(np.logaddexp(0,z)-y*z))+.5*lam*np.dot(v[:-1],v[:-1]);r=w*(expit(z)-y);grad=np.r_[x.T@r/n+lam*v[:-1],r.mean()];return loss,grad
  result=minimize(objective,np.zeros(d+1),jac=True,method='L-BFGS-B',options=dict(maxiter=self.max_iter,ftol=1e-9,gtol=1e-6,maxls=40));self.coef_=result.x[:-1];self.intercept_=result.x[-1];self.optimization_=dict(success=bool(result.success),iterations=int(result.nit),message=str(result.message),gradient_max_abs=float(abs(result.jac).max()),objective=float(result.fun));assert np.isfinite(result.x).all();return self
 def predict_proba(self,x):
  p=expit(np.asarray(x)@self.coef_+self.intercept_);return np.stack([1-p,p],1)
def balanced_accuracy_score(y,p):
 y=np.asarray(y);p=np.asarray(p);return float(np.mean([np.mean(p[y==v]==v) for v in (0,1) if np.any(y==v)]))
