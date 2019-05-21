#include <cosTaperWfld.h>
#include <cstring>
using namespace SEP;

cosTaperWfld::cosTaperWfld(const  std::shared_ptr<float4DReg> model, const  std::shared_ptr<float4DReg> data,
                            int bz, int bx, int width, float alpha, float beta) {

  assert(data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(1).n);
  assert(data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(2).n);
  assert(data->getHyper()->getAxis(3).n == model->getHyper()->getAxis(3).n);
  assert(data->getHyper()->getAxis(4).n == model->getHyper()->getAxis(4).n);

  _nx = model->getHyper()->getAxis(2).n;
  _nz = model->getHyper()->getAxis(1).n;
  _nw = model->getHyper()->getAxis(3).n;
  _nt = model->getHyper()->getAxis(4).n;

  //ensure we have five wavefields
  assert(_nw==5);

  _bz = bz;       _bx = bx;
  _alpha = alpha;
  _beta=beta;
  _rampWidthX = width;
  _rampWidthZ = width;
  _widthX=_bx+_rampWidthX;
  _widthZ=_bz+_rampWidthZ;
  _tapX.reset(new float1DReg(_widthX));
  _tapZ.reset(new float1DReg(_widthZ));
  //_nshift = (_nx-_bx-_shift)*(_nz);
  buildTaper();

  setDomainRange(model,data);

}

void cosTaperWfld::buildTaper(){

  _tapX->scale(0.0);
  _tapZ->scale(0.0);

  // Cosine padding
  for (int i=0; i<_rampWidthX; i++){
      float arg = M_PI / 2 * (_rampWidthX-i)/(_rampWidthX);
      if(cos(arg)<=0.000001) arg=0;
      else arg = _beta + pow(cos(arg),_alpha);
      (*_tapX->_mat)[_bx+i] = arg;
  }
  for (int i=0; i<_rampWidthZ; i++){
      float arg = M_PI / 2 * (_rampWidthZ-i)/(_rampWidthZ);
      if(cos(arg)<=0.000001) arg=0;
      else arg = _beta + pow(cos(arg),_alpha);
      (*_tapZ->_mat)[_bz+i] = arg;
  }

}

void cosTaperWfld::forward(const bool add,const  std::shared_ptr<float4DReg> model,  std::shared_ptr<float4DReg> data) const {

  assert(checkDomainRange(model,data));
  //if(!add) data->scale(0.);
  if(!add) std::memset(data->getVals(), 0, (long long)4 * data->getHyper()->getN123());
  // data->scaleAdd(model,0.0,1.0);

  #pragma omp parallel for collapse(4)
  for(int it = 0; it < _nt; it++){
    for(int iw = 0; iw < _nw; iw++){
      for(int ix = 0; ix < _nx; ix++){
        for(int iz = 0; iz < _nz; iz++){
        (*data->_mat)[it][iw][ix][iz] += (*model->_mat)[it][iw][ix][iz];
      }}}}

  #pragma omp parallel for collapse(2)
  for(int it = 0; it < _nt; it++){
    for(int iw = 0; iw < _nw; iw++){
      for(int ix = 0; ix < _widthX; ix++){
        for(int iz = 0; iz < _nz; iz++){
        (*data->_mat)[it][iw][ix][iz] *= (*_tapX->_mat)[ix];
        (*data->_mat)[it][iw][_nx-ix-1][iz] *= (*_tapX->_mat)[ix];
      }}}}

  #pragma omp parallel for collapse(2)
  for(int it = 0; it < _nt; it++){
    for(int iw = 0; iw < _nw; iw++){
      for(int ix = 0; ix < _nx; ix++){
        for(int iz = 0; iz < _widthZ; iz++){
        (*data->_mat)[it][iw][ix][iz] *=(*_tapZ->_mat)[iz];
        (*data->_mat)[it][iw][ix][_nz-iz-1] *=(*_tapZ->_mat)[iz];
      }}}}

}

void cosTaperWfld::adjoint(const bool add, std::shared_ptr<float4DReg> model,  const std::shared_ptr<float4DReg> data) const {

  assert(checkDomainRange(model,data));
  // if(!add) model->scale(0.);
  if(!add) std::memset(model->getVals(), 0, (long long)4 * model->getHyper()->getN123());
  //model->scaleAdd(data,0.0,1.0);

  #pragma omp parallel for collapse(4)
  for(int it = 0; it < _nt; it++){
    for(int iw = 0; iw < _nw; iw++){
      for(int ix = 0; ix < _nx; ix++){
        for(int iz = 0; iz < _nz; iz++){
        (*model->_mat)[it][iw][ix][iz] += (*data->_mat)[it][iw][ix][iz];
      }}}}

  #pragma omp parallel for collapse(2)
  for(int it = 0; it < _nt; it++){
    for(int iw = 0; iw < _nw; iw++){
      for(int ix = 0; ix < _widthX; ix++){
        for(int iz = 0; iz < _nz; iz++){
        (*model->_mat)[it][iw][ix][iz] *= (*_tapX->_mat)[ix];
        (*model->_mat)[it][iw][_nx-ix-1][iz] *= (*_tapX->_mat)[ix];
      }}}}

  #pragma omp parallel for collapse(2)
  for(int it = 0; it < _nt; it++){
    for(int iw = 0; iw < _nw; iw++){
      for(int ix = 0; ix < _nx; ix++){
        for(int iz = 0; iz < _widthZ; iz++){
        (*model->_mat)[it][iw][ix][iz] *= (*_tapZ->_mat)[iz];
        (*model->_mat)[it][iw][ix][_nz-iz-1] *= (*_tapZ->_mat)[iz];
      }}}}


}
