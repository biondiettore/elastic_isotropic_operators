#include <staggerWfld.h>
using namespace SEP;

staggerWfld::staggerWfld(const  std::shared_ptr<double4DReg> model, const  std::shared_ptr<double4DReg> data) {

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

  setDomainRange(model,data);

}

void staggerWfld::forward(const bool add,const  std::shared_ptr<double4DReg> model,  std::shared_ptr<double4DReg> data) const {

  assert(checkDomainRange(model,data));
  if(!add) data->scale(0.);


  #pragma omp parallel for 
  for(int it = 0; it < _nt; it++){
    for(int ix = 0; ix < _nx; ix++){
      for(int iz = 0; iz < _nz; iz++){
        (*data->_mat)[it][2][ix][iz] += (*model->_mat)[it][2][ix][iz];
        (*data->_mat)[it][3][ix][iz] += (*model->_mat)[it][3][ix][iz];

        //center
        if(ix < _nx-1 && iz < _nz-1){
          (*data->_mat)[it][0][ix][iz] += 0.5 * ((*model->_mat)[it][0][ix][iz] + (*model->_mat)[it][0][ix+1][iz] ); //shift right
          (*data->_mat)[it][1][ix][iz] += 0.5 * ((*model->_mat)[it][1][ix][iz] + (*model->_mat)[it][1][ix][iz+1] ); //shift down
          (*data->_mat)[it][4][ix][iz] += 0.25 * ((*model->_mat)[it][4][ix][iz] + (*model->_mat)[it][4][ix][iz+1] + (*model->_mat)[it][4][ix+1][iz] + (*model->_mat)[it][4][ix+1][iz+1] ); //shift right and down
        }
        //right side
        else if(ix == _nx-1 && iz < _nz-1){
          (*data->_mat)[it][0][_nx-1][iz] += (*model->_mat)[it][0][_nx-1][iz];
          (*data->_mat)[it][1][_nx-1][iz] += 0.5 * ((*model->_mat)[it][1][_nx-1][iz] + (*model->_mat)[it][1][_nx-1][iz+1] );
          (*data->_mat)[it][4][_nx-1][iz] += 0.5 * ((*model->_mat)[it][4][_nx-1][iz] + (*model->_mat)[it][4][_nx-1][iz+1]);
        }
        //bottom side
        else if(ix < _nx-1 && iz == _nz-1){
          (*data->_mat)[it][0][ix][_nz-1] += 0.5 * ((*model->_mat)[it][0][ix][_nz-1] + (*model->_mat)[it][0][ix+1][_nz-1] );
          (*data->_mat)[it][1][ix][_nz-1] += (*model->_mat)[it][1][ix][_nz-1];
          (*data->_mat)[it][4][ix][_nz-1] += 0.5 * ((*model->_mat)[it][4][ix][_nz-1] + (*model->_mat)[it][4][ix+1][_nz-1]);
        }
        //right corner
        else if (ix == _nx-1 && iz == _nz-1){
          (*data->_mat)[it][0][_nx-1][_nz-1] += (*model->_mat)[it][0][_nx-1][_nz-1];
          (*data->_mat)[it][1][_nx-1][_nz-1] += (*model->_mat)[it][1][_nx-1][_nz-1];
          (*data->_mat)[it][4][_nx-1][_nz-1] += (*model->_mat)[it][4][_nx-1][_nz-1];
        }
      }
    }
  }
  //
  // #pragma omp parallel for collapse(3)
  // for(int it = 0; it < _nt; it++){
  //   for(int ix = 0; ix < _nx; ix++){
  //     for(int iz = 0; iz < _nz; iz++){
  //       (*data->_mat)[it][2][ix][iz] += (*model->_mat)[it][2][ix][iz];
  //       (*data->_mat)[it][3][ix][iz] += (*model->_mat)[it][3][ix][iz];
  //     }
  //   }
  // }
  //
  // //spread center
  // #pragma omp parallel for collapse(3)
  // for(int it = 0; it < _nt; it++){
  //   for(int ix = 0; ix < _nx-1; ix++){
  //     for(int iz = 0; iz < _nz-1; iz++){
  //       (*data->_mat)[it][0][ix][iz] += 0.5 * ((*model->_mat)[it][0][ix][iz] + (*model->_mat)[it][0][ix+1][iz] ); //shift right
  //       (*data->_mat)[it][1][ix][iz] += 0.5 * ((*model->_mat)[it][1][ix][iz] + (*model->_mat)[it][1][ix][iz+1] ); //shift down
  //       (*data->_mat)[it][4][ix][iz] += 0.25 * ((*model->_mat)[it][4][ix][iz] + (*model->_mat)[it][4][ix][iz+1] + (*model->_mat)[it][4][ix+1][iz] + (*model->_mat)[it][4][ix+1][iz+1] ); //shift right and down
  //     }
  //   }
  // }
  //
  // //handle grid boundary on right side
  // #pragma omp parallel for collapse(2)
  // for(int it = 0; it < _nt; it++){
  //   for(int iz = 0; iz < _nz-1; iz++){
  //     (*data->_mat)[it][0][_nx-1][iz] += (*model->_mat)[it][0][_nx-1][iz];
  //     (*data->_mat)[it][1][_nx-1][iz] += 0.5 * ((*model->_mat)[it][1][_nx-1][iz] + (*model->_mat)[it][1][_nx-1][iz+1] );
  //     (*data->_mat)[it][4][_nx-1][iz] += 0.5 * ((*model->_mat)[it][4][_nx-1][iz] + (*model->_mat)[it][4][_nx-1][iz+1]);
  //   }
  // }
  //
  // //handle grid boundary on bottom side
  // #pragma omp parallel for collapse(2)
  // for(int it = 0; it < _nt; it++){
  //   for(int ix = 0; ix < _nx-1; ix++){
  //     (*data->_mat)[it][0][ix][_nz-1] += 0.5 * ((*model->_mat)[it][0][ix][_nz-1] + (*model->_mat)[it][0][ix+1][_nz-1] );
  //     (*data->_mat)[it][1][ix][_nz-1] += (*model->_mat)[it][1][ix][_nz-1];
  //     (*data->_mat)[it][4][ix][_nz-1] += 0.5 * ((*model->_mat)[it][4][ix][_nz-1] + (*model->_mat)[it][4][ix+1][_nz-1]);
  //   }
  // }
  //
  // //handle bottom right corner
  // #pragma omp parallel for
  // for(int it = 0; it < _nt; it++){
  //   (*data->_mat)[it][0][_nx-1][_nz-1] += (*model->_mat)[it][0][_nx-1][_nz-1];
  //   (*data->_mat)[it][1][_nx-1][_nz-1] += (*model->_mat)[it][1][_nx-1][_nz-1];
  //   (*data->_mat)[it][4][_nx-1][_nz-1] += (*model->_mat)[it][4][_nx-1][_nz-1];
  // }
}

void staggerWfld::adjoint(const bool add, std::shared_ptr<double4DReg> model,  const std::shared_ptr<double4DReg> data) const {

  assert(checkDomainRange(model,data));
  if(!add) model->scale(0.);

  //#pragma omp parallel for collapse(3)
  for(int it = 0; it < _nt; it++){
    for(int ix = 0; ix < _nx; ix++){
      for(int iz = 0; iz < _nz; iz++){
        (*model->_mat)[it][2][ix][iz] += (*data->_mat)[it][2][ix][iz];
        (*model->_mat)[it][3][ix][iz] += (*data->_mat)[it][3][ix][iz];
      }
    }
  }

  //#pragma omp parallel for collapse(3)
  for(int it = 0; it < _nt; it++){
    for(int ix = 0; ix < _nx-1; ix++){
      for(int iz = 0; iz < _nz-1; iz++){
        double pointToSpread_x = 0.5 * (*data->_mat)[it][0][ix][iz];
        (*model->_mat)[it][0][ix][iz] += pointToSpread_x;
        (*model->_mat)[it][0][ix+1][iz] += pointToSpread_x;

        double pointToSpread_z = 0.5 * (*data->_mat)[it][1][ix][iz];
        (*model->_mat)[it][1][ix][iz] += pointToSpread_z;
        (*model->_mat)[it][1][ix][iz+1] += pointToSpread_z;

        double pointToSpread_xz = 0.25 * (*data->_mat)[it][4][ix][iz];
        (*model->_mat)[it][4][ix][iz] += pointToSpread_xz;
        (*model->_mat)[it][4][ix][iz+1] += pointToSpread_xz;
        (*model->_mat)[it][4][ix+1][iz] += pointToSpread_xz;
        (*model->_mat)[it][4][ix+1][iz+1] += pointToSpread_xz;
      }
    }
  }

  //handle grid boundary on right side
  #pragma omp parallel for collapse(1)
  for(int it = 0; it < _nt; it++){
    for(int iz = 0; iz < _nz-1; iz++){
      double pointToSpread_x = (*data->_mat)[it][0][_nx-1][iz];
      (*model->_mat)[it][0][_nx-1][iz] += pointToSpread_x;

      double pointToSpread_z = 0.5 * (*data->_mat)[it][1][_nx-1][iz];
      (*model->_mat)[it][1][_nx-1][iz] += pointToSpread_z;
      (*model->_mat)[it][1][_nx-1][iz+1] += pointToSpread_z;

      double pointToSpread_xz = 0.5 * (*data->_mat)[it][4][_nx-1][iz];
      (*model->_mat)[it][4][_nx-1][iz] += pointToSpread_xz;
      (*model->_mat)[it][4][_nx-1][iz+1] += pointToSpread_xz;
    }
  }

  //handle grid boundary on bottom side
  #pragma omp parallel for collapse(1)
  for(int it = 0; it < _nt; it++){
    for(int ix = 0; ix < _nx-1; ix++){
      double pointToSpread_x = 0.5 * (*data->_mat)[it][0][ix][_nz-1];
      (*model->_mat)[it][0][ix][_nz-1] += pointToSpread_x;
      (*model->_mat)[it][0][ix+1][_nz-1] += pointToSpread_x;

      double pointToSpread_z = (*data->_mat)[it][1][ix][_nz-1];
      (*model->_mat)[it][1][ix][_nz-1] += pointToSpread_z;

      double pointToSpread_xz = 0.5 * (*data->_mat)[it][4][ix][_nz-1];
      (*model->_mat)[it][4][ix][_nz-1] += pointToSpread_xz;
      (*model->_mat)[it][4][ix+1][_nz-1] += pointToSpread_xz;
    }
  }

  //handle bottom right corner
  #pragma omp parallel for
  for(int it = 0; it < _nt; it++){
    (*model->_mat)[it][0][_nx-1][_nz-1] += (*data->_mat)[it][0][_nx-1][_nz-1];
    (*model->_mat)[it][1][_nx-1][_nz-1] += (*data->_mat)[it][1][_nx-1][_nz-1];
    (*model->_mat)[it][4][_nx-1][_nz-1] += (*data->_mat)[it][4][_nx-1][_nz-1];
  }

}
