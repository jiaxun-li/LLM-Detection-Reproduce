"""Analytic clean statistics; no enumeration of the vocabulary^K state space."""
import numpy as np


def audit(p, q, score):
    means=(p*score).sum(axis=1)
    low,high=float(means.min()),float(means.max())
    mu=(low+high)/2; delta=(high-low)/2
    b=float(score.min()); eta=float(score.min(axis=1).max()-b)
    human=float((q*score).sum()/2)
    result=dict(mu0=human,mu1=mu,delta=delta,eta=eta,b=b,
                mean_min=low,mean_max=high,mean_ordering=bool(b+eta<human<low))
    limit=min(low-b,float(np.ptp(score)))
    if not result['mean_ordering'] or limit<=0:
        return dict(result,certified=False,strong_condition=False,r=None,rho=None,
                    epsilon_raw=None,epsilon_clip=None,strong_margin=None)
    lower=(high-human)/(high-b-eta)
    cap=np.nextafter(limit,0.)
    gaps=(score-b).ravel()
    candidates=np.unique(np.r_[np.linspace(cap/1500,cap,1500),gaps[(gaps>0)&(gaps<limit)]])
    options=[]
    for r in candidates:
        rho=float((q*((score-b)<r)).sum()/2)
        upper=(low-human-rho*r)/(low-b-r)
        strong=(mu-human)/(mu-b)-rho-(2*delta+eta)/r
        options.append((min(upper,1.)-lower,float(r),rho,float(upper),float(strong)))
    width,r,rho,upper,strong=max(options,key=lambda x:x[0])
    return dict(result,r=r,rho=rho,epsilon_raw=lower,epsilon_clip=upper,
                strong_margin=strong,best_strong_margin=max(x[-1] for x in options),
                certified=bool(width>1e-10),strong_condition=bool(strong>1e-10))


def floor_for(a, epsilon):
    d=a['mu0']-(1-epsilon)*a['mean_min']-epsilon*a['b']
    tmin=d/(epsilon-a['rho'])
    assert 0<tmin<a['r']
    return a['b']+(tmin+a['r'])/2


def analytic_attack_mean(p,q,score,epsilon,K,attack):
    """Exact stationary means for specified attacks, not a worst-policy claim."""
    if attack=='minimum_zero':
        # Token bits iid with P(bit=1)=(1-epsilon)/2; parity of K bits.
        w1=(1-epsilon**K)/2
    elif attack=='minimum_force_zero':
        # Every clean step resets parity to a fair bit, each attack forces zero.
        w1=(1-epsilon)/2
    elif attack=='human_mixture':
        w1=.5
    else:
        raise ValueError(attack)
    target=(p*score).sum(axis=1)
    bad=(q*score).sum(axis=1) if attack=='human_mixture' else score.min(axis=1)
    reward=(1-epsilon)*target+epsilon*bad
    return float(np.array([1-w1,w1])@reward)
