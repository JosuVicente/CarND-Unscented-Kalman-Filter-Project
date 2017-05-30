#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {

	///* initially set to false, set to true in first call of ProcessMeasurement
	is_initialized_ = false;

	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);
	x_.fill(0);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);
	P_ << 0.2, 0, 0, 0, 0,
		0, 0.2, 0, 0, 0,
		0, 0, 0.2, 0, 0,
		0, 0, 0, 0.3, 0,
		0, 0, 0, 0, 0.3;

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 1;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.6;

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	///* State dimension
	n_x_ = 5;

	///* Augmented state dimension
	n_aug_ = 7;

	///* Sigma point spreading parameter
	lambda_ = 3 - n_x_;

	///* Set weights
	weights_ = VectorXd(2 * n_aug_ + 1);
	weights_.fill(0.5 / (lambda_ + n_aug_));
	weights_(0) = lambda_ / (lambda_ + n_aug_);
}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_)
		|| (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_)) {
		return;
	}

	/*****************************************************************************
	*  Initialization
	****************************************************************************/
	if (!is_initialized_) {
		/**
		* Initialize the state ekf_.x_ with the first measurement.
		**/

		// first measurement
		cout << "UKF: " << endl;

		//first timestamp
		time_us_ = meas_package.timestamp_;

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			x_(0) = meas_package.raw_measurements_(0)*cos(meas_package.raw_measurements_(1));
			x_(1) = meas_package.raw_measurements_(0)*sin(meas_package.raw_measurements_(1));
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			x_(0) = meas_package.raw_measurements_(0);
			x_(1) = meas_package.raw_measurements_(1);
		}

		// done initializing, no need to predict or update
		is_initialized_ = true;

		//Initialize NIS vectors
		NIS_LASER_ = VectorXd(1);
		NIS_RADAR_ = VectorXd(1);
		return;
	}
	else {
		/*****************************************************************************
		*  Prediction
		****************************************************************************/

		/**
		Compute the time elapsed between the current and previous measurements */
		double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
		time_us_ = meas_package.timestamp_;

		Prediction(dt);

		/*****************************************************************************
		*  Update
		****************************************************************************/

		/**
		* Use the sensor type to perform the update step.
		*/

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			// Radar updates
			UpdateRadar(meas_package);
		}
		else {
			// Laser updates	  
			UpdateLidar(meas_package);
		}
	}

}
/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t) {
	/**
	Complete this function! Estimate the object's location. Modify the state
	vector, x_. Predict sigma points, the state, and the state covariance matrix.
	*/

	/* Sigma Points .*/
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	AugmentedSigmaPoints(&Xsig_aug);

	/* Predict Sigma Points .*/
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
	SigmaPointPrediction(&Xsig_pred_, Xsig_aug, delta_t);

	/* Predict Mean and Covariance  .*/
	PredictMeanAndCovariance();
}

/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
	TODO:

	Complete this function! Use lidar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.

	You'll also need to calculate the lidar NIS.
	*/
	//std::cout << "******** UPDATE LIDAR *************" << std::endl;

	VectorXd z = VectorXd(2);
	MatrixXd Zsig = MatrixXd(2, 2 * n_aug_ + 1);
	MatrixXd S = MatrixXd(2, 2);

	//Predict Lidar
	PredictLidarMeasurement(&z, &Zsig, &S);

	//Update State
	UpdateState(z, Zsig, S, meas_package.raw_measurements_, meas_package.sensor_type_);
}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**
	TODO:

	Complete this function! Use radar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.

	You'll also need to calculate the radar NIS.
	*/
	//std::cout << "******** UPDATE RADAR *************" << std::endl;
	VectorXd z = VectorXd(3);
	MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);
	MatrixXd S = MatrixXd(3, 3);

	//Predict Radar
	PredictRadarMeasurement(&z, &Zsig, &S);

	//Update State
	UpdateState(z, Zsig, S, meas_package.raw_measurements_, meas_package.sensor_type_);
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

	//create augmented mean vector
	VectorXd x_aug = VectorXd(7);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(7, 7);

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	x_aug.fill(0);
	x_aug.head(5) = x_;

	//create augmented covariance matrix
	P_aug.fill(0);
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = pow(std_a_, 2);
	P_aug(6, 6) = pow(std_yawdd_, 2);

	//create square root matrix
	MatrixXd A = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 1; i <= n_aug_; i++) {
		Xsig_aug.col(i) = x_aug + sqrt(lambda_ + n_aug_)*A.col(i - 1);
		Xsig_aug.col(n_aug_ + i) = x_aug - +sqrt(lambda_ + n_aug_)*A.col(i - 1);
	}

	*Xsig_out = Xsig_aug;


}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out, MatrixXd& Xsig_aug, double delta_t) {

	//create matrix with predicted sigma points as columns
	MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

	//predict sigma points
	//avoid division by zero
	//write predicted sigma points into right column
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		double px = Xsig_aug(0, i);
		double py = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawrate = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yaw = Xsig_aug(6, i);

		double px_o = px;
		double py_o = py;

		if (fabs(yawrate) > LOW_V) {
			px_o += (v / yawrate)*(sin(yaw + yawrate*delta_t) - sin(yaw));
			py_o += (v / yawrate)*(-cos(yaw + yawrate*delta_t) + cos(yaw));
		}
		else {
			px_o += v*cos(yaw)*delta_t;
			py_o += v*sin(yaw)*delta_t;
		}

		px_o += 0.5*pow(delta_t, 2)*cos(yaw)*nu_a;
		py_o += 0.5*pow(delta_t, 2)*sin(yaw)*nu_a;

		double v_o = v + delta_t * nu_a;
		double yaw_o = yaw + (yawrate*delta_t) + 0.5 *pow(delta_t, 2) * nu_yaw;
		double yawrate_o = yawrate + delta_t * nu_yaw;



		Xsig_pred(0, i) = px_o;
		Xsig_pred(1, i) = py_o;
		Xsig_pred(2, i) = v_o;
		Xsig_pred(3, i) = yaw_o;
		Xsig_pred(4, i) = yawrate_o;

	}

	//write result
	*Xsig_out = Xsig_pred;

}

void UKF::PredictMeanAndCovariance() {

	//predict state mean
	x_.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		x_ += weights_(i) * Xsig_pred_.col(i);
	}

	//predict state covariance matrix
	P_.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
												// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}

}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* Zsig_out, MatrixXd* S_out) {

	//set measurement dimension, radar can measure r, phi, and r_dot
	int n_z = 3;

	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);

	//transform sigma points into measurement space
	Zsig.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		float px = Xsig_pred_(0, i);
		float py = Xsig_pred_(1, i);
		float v = Xsig_pred_(2, i);
		float yaw = Xsig_pred_(3, i);
		float yawrate = Xsig_pred_(4, i);

		Zsig(0, i) = sqrt(pow(px, 2) + pow(py, 2));
		Zsig(1, i) = atan2(py, px);
		Zsig(2, i) = (px*cos(yaw)*v + py*sin(yaw)*v) / Zsig(0, i);

	}

	//calculate mean predicted measurement
	z_pred.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred += weights_(i)*Zsig.col(i);
	}

	//calculate measurement covariance matrix S
	S.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	MatrixXd R = MatrixXd(n_z, n_z);
	R.fill(0);
	R(0, 0) = std_radr_*std_radr_;
	R(1, 1) = std_radphi_*std_radphi_;
	R(2, 2) = std_radrd_*std_radrd_;

	S += R;

	//write result
	*z_out = z_pred;
	*S_out = S;
	*Zsig_out = Zsig;
}

void UKF::PredictLidarMeasurement(VectorXd* z_out, MatrixXd* Zsig_out, MatrixXd* S_out) {

	//set measurement dimension, radar can measure r, phi, and r_dot
	int n_z = 2;

	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);

	//transform sigma points into measurement space
	Zsig.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		float px = Xsig_pred_(0, i);
		float py = Xsig_pred_(1, i);

		Zsig(0, i) = px;
		Zsig(1, i) = py;

	}

	//calculate mean predicted measurement
	z_pred.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred += weights_(i)*Zsig.col(i);
	}

	//calculate measurement covariance matrix S
	S.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Zsig.col(i) - z_pred;
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	MatrixXd R = MatrixXd(n_z, n_z);
	R.fill(0);
	R(0, 0) = std_laspx_*std_laspx_;
	R(1, 1) = std_laspy_*std_laspy_;

	S += R;

	//write result
	*z_out = z_pred;
	*S_out = S;
	*Zsig_out = Zsig;
}

void UKF::UpdateState(VectorXd& z_pred, MatrixXd& Zsig, MatrixXd& S, VectorXd z, MeasurementPackage::SensorType sensor_type) {

	//set measurement dimension, radar can measure r, phi, and r_dot
	int n_z = z_pred.size();

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization

		/*if (n_z == 3) {
		x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));
		z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));
		}*/
		x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));
		z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

		Tc += weights_(i) * x_diff * z_diff.transpose();
	}

	//calculate Kalman gain K;
	MatrixXd K = Tc*S.inverse();

	//update state mean and covariance matrix
	VectorXd z_diff = z - z_pred;
	z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));
	x_ += K*z_diff;

	P_ -= K*S*K.transpose();

	//Calculate and store NIS
	float NIS = (z_diff.transpose())*S.inverse()*z_diff;

	if (sensor_type == MeasurementPackage::RADAR) {
		NIS_RADAR_[NIS_LASER_.size() - 1] = NIS;
		NIS_RADAR_.conservativeResize(NIS_LASER_.size() + 1);

	}
	else {
		NIS_LASER_[NIS_LASER_.size() - 1] = NIS;
		NIS_LASER_.conservativeResize(NIS_LASER_.size() + 1);
	}

	

}
