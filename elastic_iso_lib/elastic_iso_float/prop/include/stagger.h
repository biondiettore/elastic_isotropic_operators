#ifndef STAGGER_H
#define STAGGER_H 1

#include <float2DReg.h>
#include <float4DReg.h>
#include <operator.h>

using namespace SEP;
//! Interpolate a grid to the left or right one half grid cell
/*!
 Used for creating the staggered grid elastic parameters used for leastic wave prop
*/
class staggerX : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:

    int _nx;
    int _nz;

	public:
    //! Constructor.
		/*!
    * Overloaded constructors from operator
    */
		staggerX(const std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float2DReg> data);

    //! FWD
		/*!
    * this interpolates a grid of values 1/2 grid point to the right
    */
    void forward(const bool add, const std::shared_ptr<SEP::float2DReg> model, std::shared_ptr<SEP::float2DReg> data) const;

    //! ADJ
    /*!
    * this interpolates a grid of values 1/2 grid point to the left
    */
		void adjoint(const bool add, std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float2DReg> data) const;

		//! Desctructor
    /*!
    * A more elaborate description of Desctructor
    */
		~staggerX(){};

};

//! Interpolate a grid of values up or down one half grid cell
/*!
 Used for creating the staggered grid elastic parameters used for leastic wave prop
*/
class staggerZ : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:

    int _nx;
    int _nz;

	public:
    //! Constructor.
		/*!
    * Overloaded constructors from operator
    */
		staggerZ(const std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float2DReg> data);

    //! FWD
		/*!
    * this interpolates a grid of values 1/2 grid point down
    */
    void forward(const bool add, const std::shared_ptr<SEP::float2DReg> model, std::shared_ptr<SEP::float2DReg> data) const;

    //! ADJ
    /*!
    * this interpolates a grid of values 1/2 grid point up
    */
		void adjoint(const bool add, std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float2DReg> data) const;

		//! Desctructor
    /*!
    * A more elaborate description of Desctructor
    */
		~staggerZ(){};

};
//
// //! Interpolate a grid of values up or down one half grid cell
// /*!
//  Used for creating the staggered elastic wavefield used for elastic wave eq. Expected input dimensions are (nz,nx,nw,nt) where nz is the fast axis
// */
// class staggerWfld : public Operator<SEP::float4DReg, SEP::float4DReg> {
//
// 	private:
//
//     int _nx;
//     int _nz;
// 		int _nt;
// 		int _nw; //number of wavefields
//
// 	public:
//     //! Constructor.
// 		/*!
//     * Overloaded constructors from operator
//     */
// 		staggerWfld(const std::shared_ptr<float4DReg> model, const std::shared_ptr<float4DReg> data);
//
//     //! FWD
// 		/*!
//     * this interpolates the: first wavefield left 1/2 cell, second wavefield down 1/2 cell, and the fifth wavefield down 1/2 cell and left 1/2 cell
//     */
//     void forward(const bool add, const std::shared_ptr<float4DReg> model, std::shared_ptr<float4DReg> data) const;
//
//     //! ADJ
//     /*!
//     * this interpolates the: first wavefield right 1/2 cell, second wavefield up 1/2 cell, and the fifth wavefield up 1/2 cell and right 1/2 cell
//     */
// 		void adjoint(const bool add, std::shared_ptr<float4DReg> model, const std::shared_ptr<float4DReg> data) const;
//
// 		//! Desctructor
//     /*!
//     * A more elaborate description of Desctructor
//     */
// 		~staggerWfld(){};
//
// };
//
#endif
