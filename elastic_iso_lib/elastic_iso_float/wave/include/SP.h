#ifndef SP_H
#define SP_H 1


#include <float3DReg.h>
#include <float4DReg.h>
#include <operator.h>

using namespace SEP;
//! Pad or truncate to/from source space to wavefield on staggered grid
/*!
 Used for creating a wavefield (z,x,t,w) from a source function (t,w,x,z) or functions, and staggering the resulting wavefield
*/
class SP : public Operator<float3DReg, float4DReg> {

	private:

    int _nx_model, _nx_data;
    int _nz_model, _nz_data;
    int _ox_model, _ox_data;
    int _oz_model, _oz_data;
    int _dx_model, _dx_data;
    int _dz_model, _dz_data;
    int _nt;
    int _nw; //number of wavefields
		std::vector<int> _gridPointIndexUnique; /** Array containing all the positions of the excited grid points - each grid point is unique */
		std::shared_ptr<float4DReg> _tempWfld;

	public:
    //! Constructor.
		/*!
    * Overloaded constructors from operator
    */
		SP(const std::shared_ptr<float3DReg> model, const std::shared_ptr<float4DReg> data, std::vector<int> gridPointIndexUnique);

    //! FWD
		/*!
    * this pads from source to wavefield
    */
    void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float4DReg> data) const;

    //! ADJ
    /*!
    * this truncates from wavefield to source
    */
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float4DReg> data) const;

		//! Desctructor
    /*!
    * A more elaborate description of Desctructor
    */
		~SP(){};

};

#endif
