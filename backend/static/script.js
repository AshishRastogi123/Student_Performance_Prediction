function hideAllSections(){
    document.getElementById("predictionSection").style.display="none";
    document.getElementById("teamSection").style.display="none";
    document.getElementById("aboutProject").style.display="none";
}

function setActive(element){
    let items=document.querySelectorAll(".sidebar ul li");
    items.forEach(item=>item.classList.remove("active"));
    element.classList.add("active");
}

function showPrediction(element){
    hideAllSections();
    document.getElementById("predictionSection").style.display="block";
    document.getElementById("pageTitle").innerText="Student Performance Prediction";
    setActive(element);
}

function showTeam(element){
    hideAllSections();
    document.getElementById("teamSection").style.display="block";
    document.getElementById("pageTitle").innerText="Team Members";
    setActive(element);
}

function showAboutProject(element){
    hideAllSections();
    document.getElementById("aboutProject").style.display="block";
    document.getElementById("pageTitle").innerText="About Project";
    setActive(element);
}

function predict(){
    // Get all form values
    const age = document.getElementById("age").value;
    const gender = document.getElementById("gender").value;
    const study_hours = document.getElementById("study_hours").value;
    const attendance = document.getElementById("attendance").value;
    const internet = document.getElementById("internet").value;
    const sleep_hours = document.getElementById("sleep_hours").value;
    const sleep_quality = document.getElementById("sleep_quality").value;
    const facility = document.getElementById("facility").value;
    const exam_diff = document.getElementById("exam_diff").value;
    const course = document.getElementById("course").value;
    const method = document.getElementById("method").value;

    // Validate all fields are filled
    if (!age || !gender || !study_hours || !attendance || internet === "" || !sleep_hours || !sleep_quality || !facility || !exam_diff || !course || !method) {
        document.getElementById("result").innerHTML = '<div class="alert alert-warning">Please fill in all fields!</div>';
        return;
    }

    // Get the button element and show loading state
    const predictBtn = document.querySelector("button[onclick='predict()']");
    const originalBtnHTML = predictBtn.innerHTML;
    predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Processing...';
    predictBtn.disabled = true;

    // Create form data
    const formData = new FormData();
    formData.append("age", age);
    formData.append("gender", gender);
    formData.append("study_hours", study_hours);
    formData.append("attendance", attendance);
    formData.append("internet", internet);
    formData.append("sleep_hours", sleep_hours);
    formData.append("sleep_quality", sleep_quality);
    formData.append("facility", facility);
    formData.append("exam_diff", exam_diff);
    formData.append("course", course);
    formData.append("method", method);

    // Send POST request to backend
    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success === false) {
            document.getElementById("result").innerHTML = `<div class="alert alert-danger"><strong>Error:</strong> ${data.error}</div>`;
            return;
        }
        
        const prediction = data.prediction;
        const confidence = data.confidence;
        const performance = data.performance;
        const emoji = data.emoji;
        const category = data.category;
        
        // Color based on performance category
        const categories = {
            0: { color: "success", text: "Excellent" },
            1: { color: "info", text: "Good" },
            2: { color: "warning", text: "Average" },
            3: { color: "danger", text: "Poor" }
        };
        
        const categoryInfo = categories[category] || categories[0];
        
        // Display result with confidence percentage and performance
        let resultHTML = `
            <div class="alert alert-light border-2 border-${categoryInfo.color}" style="border-radius: 10px; padding: 20px;">
                <h4 style="color: #333;">${emoji} <strong>${prediction}</strong></h4>
                <hr>
                <div style="margin: 15px 0;">
                    <p><strong>Confidence Score:</strong> 
                        <span style="font-size: 18px; color: #007bff;">${confidence}%</span>
                    </p>
                    <div class="progress" style="height: 25px;">
                        <div class="progress-bar bg-${categoryInfo.color}" role="progressbar" 
                             style="width: ${confidence}%;" aria-valuenow="${confidence}" 
                             aria-valuemin="0" aria-valuemax="100">
                             ${confidence}%
                        </div>
                    </div>
                </div>
                <hr>
                <p style="margin: 0;"><strong>Performance Category:</strong> 
                    <span class="badge bg-${categoryInfo.color}" style="font-size: 14px; padding: 8px 12px;">
                        ${categoryInfo.text}
                    </span>
                </p>
            </div>
        `;
        
        document.getElementById("result").innerHTML = resultHTML;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerHTML = `<div class="alert alert-danger"><strong>Error:</strong> ${error.message}</div>`;
    })
    .finally(() => {
        // Restore button to original state
        predictBtn.innerHTML = originalBtnHTML;
        predictBtn.disabled = false;
    });
}