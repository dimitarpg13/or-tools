#ifndef OR_TOOLS_FORECASTER_FORECASTER_H_
#define OR_TOOLS_FORECASTER_FORECASTER_H_
#include <vector>
#include <string>
#include <map>
#include "fourier_1d.h"

namespace operations_research {
namespace forecaster {

class Forecaster {
  public:
    enum ForecasterType {
      FourierLinear = 0,
      FourierQuadratic = 1,
      SuperResolution = 2,
    };
    enum ForecasterStatus {
      SUCCESS=0,
      MISSING_DATA=1,
      INCORRECT_DATA_FORMAT=2,
      TRUNCATED_DATA=4,
      UNKNOWN_PROBLEM_TYPE=8,
      INTERNAL_ERROR=16,
      UNSPECIFIED=32
    };

    virtual const ForecasterStatus get_status() const = 0;
    virtual DenseDataContainer<DATA_REAL_VAL_TYPE>* get_result()  = 0;

    // define Prophet-like interface for all classes implementing the Forecaster
    //
    //TODO (dpg): finish the signature e.g. arguments
    virtual bool fit(const SparseDataContainer<DATA_REAL_VAL_TYPE>& , const DATA_LEN_TYPE&, const DATA_REAL_VAL_TYPE& ) = 0;

    virtual bool predict( ) = 0;

    virtual bool predict_trend( ) = 0;

    virtual bool predict_seasonal_components( ) = 0;

    virtual bool predicitive_samples( ) = 0;

    virtual bool validate_inputs( ) = 0;

    virtual bool validate_column_name( ) = 0;

    virtual bool setup_dataframe( ) = 0;

    virtual bool initialize_scales( ) = 0;
     
    virtual bool set_changepoints( ) = 0;

    virtual bool make_holiday_features( ) = 0;

    virtual bool make_all_seasonality_features( ) = 0;

    virtual bool add_regressor( ) = 0;

    virtual bool add_seasonality( ) = 0;

    virtual bool add_country_holidays( ) = 0;

    virtual bool add_group_component( ) = 0;

    virtual bool parse_seasonality_args( ) = 0;

    virtual bool sample_posterior_predictive( ) = 0;

    virtual bool predict_uncertainty( ) = 0;

    virtual bool sample_model( ) = 0;
    //
    // Prophet-like interface for all classes implementing Forecaster

    virtual ForecasterType GetType() = 0;

    ~Forecaster();
};


} // ns: forecaster

} // ns: operations_research

#endif // OR_TOOLS_FORECASTER_FORECASTER
