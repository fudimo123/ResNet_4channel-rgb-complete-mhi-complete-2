import single_dynamic_train as st

st.MODEL_NAME = 'improved_resnet18_spp'
st.RESULT_DIR = 'fusion_result_samm_dynamic_single_improved_resnet18_spp'
st.TRANSFER_WEIGHTS_PATH = None


def main():
    print("Starting SAMM dynamic single-channel experiment: improved_resnet18_spp")
    print("Results will be saved to: fusion_result_samm_dynamic_single_improved_resnet18_spp")
    st.main()


if __name__ == '__main__':
    main()
