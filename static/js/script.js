document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById("predictionForm");
    const button = document.getElementById("predictButton");

    if (!form || !button) {
        return;
    }


    form.addEventListener("submit", function () {

        const buttonText =
            button.querySelector(".button-text");

        const buttonArrow =
            button.querySelector(".button-arrow");


        button.disabled = true;

        buttonText.textContent =
            "Calculating...";

        buttonArrow.textContent =
            "⏳";


        button.style.opacity = "0.75";

    });


    /* ==============================
       Number Input Validation
       ============================== */

    const numberInputs =
        form.querySelectorAll(
            'input[type="number"]'
        );


    numberInputs.forEach(function (input) {

        input.addEventListener(
            "input",
            function () {

                if (this.value < 0) {
                    this.value = 0;
                }

            }
        );

    });


    /* ==============================
       ATM ID Validation
       ============================== */

    const atmInput =
        document.getElementById("atm_id");


    atmInput.addEventListener(
        "input",
        function () {

            this.value =
                this.value.replace(
                    /[^0-9]/g,
                    ""
                );

        }
    );


});