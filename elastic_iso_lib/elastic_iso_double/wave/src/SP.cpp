#include <SP.h>
#include <algorithm>
#include <cstring>
using namespace SEP;


SP::SP(const std::shared_ptr<double3DReg> model, const std::shared_ptr<double4DReg> data,std::vector<int> gridPointIndexUnique){
  // assert(data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(4).n); //z axis
  // assert(data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(3).n); //x axis
  assert(data->getHyper()->getAxis(4).n == model->getHyper()->getAxis(3).n); //time axis
  assert(data->getHyper()->getAxis(3).n == model->getHyper()->getAxis(2).n); //wavefield

  //ensure number of sources/rec to inject is equal to the number of gridpoints provided
  assert(model->getHyper()->getAxis(1).n == gridPointIndexUnique.size());

  _nz_data = data->getHyper()->getAxis(1).n;
  _nx_data = data->getHyper()->getAxis(2).n;
  _oz_data = data->getHyper()->getAxis(1).o;
  _ox_data = data->getHyper()->getAxis(2).o;
  _dz_data = data->getHyper()->getAxis(1).d;
  _dx_data= data->getHyper()->getAxis(2).d;
  _nw = data->getHyper()->getAxis(3).n;
  _nt = data->getHyper()->getAxis(4).n;

  _gridPointIndexUnique = gridPointIndexUnique;

  //ensure we have five wavefields
  assert(_nw==5);

  setDomainRange(model,data);

  _tempWfld = data->clone();
}

//! FWD
/*!
* this pads from source to wavefield
*/
void SP::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double4DReg> data) const{
  assert(checkDomainRange(model,data));
  // if(!add) data->scale(0.);
  if(!add) std::memset(data->getVals(), 0, sizeof data->getVals());
  std::memset(_tempWfld->getVals(), 0, sizeof _tempWfld->getVals());
  // _tempWfld->scale(0);

  //for each device add to correct location in wavefield
  // #pragma omp parallel for
  for(int id = 0; id < _gridPointIndexUnique.size(); id++){
      int gridPoint = _gridPointIndexUnique[id];
      int ix = (int)gridPoint/_nz_data;
      int iz = (int)(gridPoint-(int)(gridPoint/_nz_data)*_nz_data);
      #pragma omp parallel for
      for(int it = 0; it < _nt; it++){
        (*_tempWfld->_mat)[it][0][ix][iz] = (*model->_mat)[it][0][id];
        (*_tempWfld->_mat)[it][1][ix][iz] = (*model->_mat)[it][1][id];
        (*_tempWfld->_mat)[it][2][ix][iz] = (*model->_mat)[it][2][id];
        (*_tempWfld->_mat)[it][3][ix][iz] = (*model->_mat)[it][3][id];
        (*_tempWfld->_mat)[it][4][ix][iz] = (*model->_mat)[it][4][id];
      }
  }
  //
  //stagger wfld once source injected
  #pragma omp parallel for
  for(int it = 0; it < _nt; it++){
    for(int ix = 0; ix < _nx_data; ix++){
      for(int iz = 0; iz < _nz_data; iz++){
        (*data->_mat)[it][2][ix][iz] += (*_tempWfld->_mat)[it][2][ix][iz];
        (*data->_mat)[it][3][ix][iz] += (*_tempWfld->_mat)[it][3][ix][iz];

        //center
        if(ix < _nx_data-1 && iz < _nz_data-1){
          (*data->_mat)[it][0][ix][iz] += 0.5 * ((*_tempWfld->_mat)[it][0][ix][iz] + (*_tempWfld->_mat)[it][0][ix+1][iz] ); //shift right
          (*data->_mat)[it][1][ix][iz] += 0.5 * ((*_tempWfld->_mat)[it][1][ix][iz] + (*_tempWfld->_mat)[it][1][ix][iz+1] ); //shift down
          (*data->_mat)[it][4][ix][iz] += 0.25 * ((*_tempWfld->_mat)[it][4][ix][iz] + (*_tempWfld->_mat)[it][4][ix][iz+1] + (*_tempWfld->_mat)[it][4][ix+1][iz] + (*_tempWfld->_mat)[it][4][ix+1][iz+1] ); //shift right and down
        }
        //right side
        else if(ix == _nx_data-1 && iz < _nz_data-1){
          (*data->_mat)[it][0][_nx_data-1][iz] += (*_tempWfld->_mat)[it][0][_nx_data-1][iz];
          (*data->_mat)[it][1][_nx_data-1][iz] += 0.5 * ((*_tempWfld->_mat)[it][1][_nx_data-1][iz] + (*_tempWfld->_mat)[it][1][_nx_data-1][iz+1] );
          (*data->_mat)[it][4][_nx_data-1][iz] += 0.5 * ((*_tempWfld->_mat)[it][4][_nx_data-1][iz] + (*_tempWfld->_mat)[it][4][_nx_data-1][iz+1]);
        }
        //bottom side
        else if(ix < _nx_data-1 && iz == _nz_data-1){
          (*data->_mat)[it][0][ix][_nz_data-1] += 0.5 * ((*_tempWfld->_mat)[it][0][ix][_nz_data-1] + (*_tempWfld->_mat)[it][0][ix+1][_nz_data-1] );
          (*data->_mat)[it][1][ix][_nz_data-1] += (*_tempWfld->_mat)[it][1][ix][_nz_data-1];
          (*data->_mat)[it][4][ix][_nz_data-1] += 0.5 * ((*_tempWfld->_mat)[it][4][ix][_nz_data-1] + (*_tempWfld->_mat)[it][4][ix+1][_nz_data-1]);
        }
        //right corner
        else if (ix == _nx_data-1 && iz == _nz_data-1){
          (*data->_mat)[it][0][_nx_data-1][_nz_data-1] += (*_tempWfld->_mat)[it][0][_nx_data-1][_nz_data-1];
          (*data->_mat)[it][1][_nx_data-1][_nz_data-1] += (*_tempWfld->_mat)[it][1][_nx_data-1][_nz_data-1];
          (*data->_mat)[it][4][_nx_data-1][_nz_data-1] += (*_tempWfld->_mat)[it][4][_nx_data-1][_nz_data-1];
        }
      }
    }
  }

}

//! ADJ
/*!
* this truncates from wavefield to source
*/
void SP::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double4DReg> data) const{
  assert(checkDomainRange(model,data));
  if(!add) model->scale(0.);

  //for each device add to correct location in wavefield
  //#pragma omp parallel for
  for(int id = 0; id < _gridPointIndexUnique.size(); id++){
      int gridPoint = _gridPointIndexUnique[id];
      int ix = (int)gridPoint/_nz_data;
      int iz = (int)(gridPoint-(int)(gridPoint/_nz_data)*_nz_data);
      #pragma omp parallel for
      for(int it = 0; it < _nt; it++){
        (*model->_mat)[it][0][id] += 0.5 * ((*data->_mat)[it][0][ix-1][iz] + (*data->_mat)[it][0][ix][iz] ); //shift left
        (*model->_mat)[it][1][id] += 0.5 * ((*data->_mat)[it][1][ix][iz-1] + (*data->_mat)[it][1][ix][iz] ); //shift up
        (*model->_mat)[it][2][id] += (*data->_mat)[it][2][ix][iz];
        (*model->_mat)[it][3][id] += (*data->_mat)[it][3][ix][iz];
        (*model->_mat)[it][4][id] += 0.25 * ((*data->_mat)[it][4][ix][iz] + (*data->_mat)[it][4][ix][iz-1] + (*data->_mat)[it][4][ix-1][iz] + (*data->_mat)[it][4][ix-1][iz-1] ); //shift left and up
        // (*model->_mat)[it][0][id] += (*data->_mat)[it][0][ix][iz]; //shift left
        // (*model->_mat)[it][1][id] += (*data->_mat)[it][1][ix][iz]; //shift up
        // (*model->_mat)[it][2][id] += (*data->_mat)[it][2][ix][iz];
        // (*model->_mat)[it][3][id] += (*data->_mat)[it][3][ix][iz];
        // (*model->_mat)[it][4][id] += (*data->_mat)[it][4][ix][iz]; //shift left and up
      }
  }

}
