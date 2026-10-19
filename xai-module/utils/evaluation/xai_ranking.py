import pandas as pd

class XaiRanking:
    @staticmethod
    def build_ranking(dataset_name, model_name, xai_results, comparisons):
        return pd.DataFrame([{
            'dataset':dataset_name, 'model':model_name, 'method':method,
            'stability':data['stability'],
            'stability_std':data['stability_details']['global_summary']['std'],
            'top_masking_drop':data['fidelity']['fidelity_drop'],
            'top_minus_random':data['fidelity']['top_minus_random'],
            'top_minus_bottom':data['fidelity']['top_minus_bottom'],
            'runtime':data['runtime']}
            for method,data in xai_results.items()])
