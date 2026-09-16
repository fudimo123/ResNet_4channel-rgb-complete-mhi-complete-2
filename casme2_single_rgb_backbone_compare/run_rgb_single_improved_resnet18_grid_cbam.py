import single_rgb_train as st

st.MODEL_NAME = 'improved_resnet18_grid_cbam'
st.RESULT_DIR = 'fusion_result_casme2_rgb_single_improved_resnet18_grid_cbam'
st.TRANSFER_WEIGHTS_PATH = None


def main():
    print("Starting CASME2 RGB single-channel experiment: improved_resnet18_grid_cbam")
    print("Results will be saved to: fusion_result_casme2_rgb_single_improved_resnet18_grid_cbam")
    st.main()


if __name__ == '__main__':
    main()
