
var app = angular.module('myApp', []);

// enable URL for file download
app.config(['$compileProvider',
    function ($compileProvider) {
        $compileProvider.aHrefSanitizationWhitelist(/^\s*(https?|ftp|mailto|tel|file|blob):/);
    }]);

app.directive('fileModel', ['$parse', function ($parse) {
    return {
        restrict: 'A',
        link: function (scope, element, attrs) {
            var model = $parse(attrs.fileModel);
            var modelSetter = model.assign;
            element.bind('change', function () {
                scope.$apply(function () {
                    modelSetter(scope, element[0].files[0]);
                });
            });
        }
    };
}]);

app.controller('populateCtrl', function ($scope, $window, $http, $q) {
    // Initialize values
    $scope.selectResults = ['Switch', 'Model']
    // $scope.selectedModel = "dynamic";
    $scope.zipFolderName = "";
    $scope.evaluationMessage = "";
    $scope.evalErrorMessage = "";
    $scope.validEntries = false;
    $scope.evaluationSuccess = false;
    $scope.inProgress = false;
    $scope.isLoadingResults = false;
    $scope.isLoadingEvaluation = false;

    $scope.selectedOption = ''; // Initialize the selected option
    $scope.selectedSwitch = ''; // Initialize the selected switch
    $scope.showUploadSection = false; // Initialize the flag for showing the upload section
    $scope.showCheckDataButton = false; // Initialize the flag for showing the "Check Data" button
    $scope.showResultsButton = false; // Initialize the flag for showing the "Get Results" button
    $scope.OOVchoice = ''; // Initialize the selected OOV choice

    $scope.showPopup = function() {
        var confirmed = window.confirm("This option will allow you to implement computational search models on your VFT data. You will be redirected to a Google Colab notebook for the same, where you will upload data and select which models you want to examine. Do you want to go to the Colab notebook?");
        if (confirmed) {
            // Open the link in a new tab
            window.open('https://colab.research.google.com/drive/1P4ARz2h9Bf4k4XC7T59_jSpvGZQ5cKfe?usp=sharing', '_blank'); // Replace with your actual external link
        }
    };


    $scope.selectOption = function(option) {
        $scope.selectedOption = option;
        // Reset other variables if needed
        $scope.selectedSwitch = '';
        $scope.showUploadSection = false;
        $scope.showCheckDataButton = false;
        $scope.showResultsButton = false;

        if ($scope.selectedOption === 'get-models') {
            // Redirect to the desired link for "get-models" option
            window.open('https://colab.research.google.com/drive/1P4ARz2h9Bf4k4XC7T59_jSpvGZQ5cKfe?usp=sharing', '_blank');
        }

        else{
        // Update the visibility of the file upload section based on selected option
        $scope.showUploadSection = ($scope.selectedOption === 'get-sims') || 
                    ($scope.selectedOption === 'get-switch' && $scope.selectedSwitch !== '');
        }

    };


    // Evaluate data on button press
    $scope.evaluateDataButton = function (oov_choice) {
        $scope.isLoadingEvaluation = true;
        $scope.OOVchoice = oov_choice;
        if (typeof $scope.userFile === "undefined") {
            $scope.evalErrorMessage = "Please select a file.";
            $scope.evaluationSuccess = false;
            //$scope.isLoadingEvaluation = false;
        }
        else {
            $scope.evalErrorMessage = "";
            var file = $scope.userFile;
            $scope.zipFolderName = [file.name.split('.')[0], 'forager'].join('_');
            var response = $scope.retrieveDataEvaluation($scope.userFile, oov_choice);
            $scope.evaluationSuccess = true;
            //$scope.isLoadingEvaluation = false;
        }
    }

    // Retrieve results on button press
    $scope.retrieveSimsButton = function () {
        var file = $scope.userFile;

        $scope.validateEntries();

        if ($scope.validEntries)  {
            $scope.inProgress = true;
            console.log($scope.inProgress)
            $scope.evalErrorMessage = "";

            // generate results folder name

            $scope.zipFolderName = [file.name.split('.')[0], 'forager'].join('_');

            // make HTTP Request
            $scope.retrieveSimilarityResults(file);
        }
    };

    $scope.retrieveResultsButton = function(selectedOption, selectedSwitch){
        // Set loading state to true when button is clicked
        $scope.isLoadingResults = true;
        var file = $scope.userFile;

        $scope.validateEntries();

        if ($scope.validEntries)  {
            $scope.inProgress = true;
            console.log($scope.inProgress)
            $scope.evalErrorMessage = "";

            $scope.zipFolderName = [file.name.split('.')[0], 'forager'].join('_');

            // make HTTP Request

            console.log("selectedSwitch: " + selectedSwitch)

            $scope.retrieveResults(file, $scope.selectedOption, $scope.selectedSwitch);

            
        }
    };

    // Ensure form has valid entries before model request
    $scope.validateEntries = function(){
        $scope.validEntries = false;
        if (typeof $scope.userFile === "undefined") {
            $scope.resultsErrorMessage = "Please upload a file.";
        } 
        //else if ($scope.getSwitch != true){// && $scope.getModel != true && $scope.getNll != true) {
        //    $scope.resultsErrorMessage = "Please select at least one output.";
        //} 
        else {
            $scope.validEntries = true;
        }
    }

    // HTTP request to evaluate data
    $scope.retrieveDataEvaluation = function (file, oov_choice) {
        var payload = new FormData();
        payload.append('filename', file);
        payload.append('selected-oov', oov_choice);

        var requestBody = {
            url: 'evaluate-data',
            method: 'POST',
            data: payload,
            headers: { 'Content-Type': undefined },
            transformRequest: angular.identity,
            responseType: 'json'
        }
        
        $http(requestBody).then(function (response) {
            // Parse the JSON response
            var responseData = response.data;

            

            // Update the message to display
            $scope.evaluationMessage = responseData.message;
            
            $scope.showResultsButton = true;
            $scope.evaluationSuccess = true;

            // Show the download button
            $scope.showDownloadButton = true;
            $scope.isLoadingEvaluation = false;


            $scope.zipContent = responseData.zipContent;

            $scope.resultsErrorMessageStyle = {
                    color: 'red',
                    fontWeight: 'bold'
                };
                
            }).catch(function (data) {
            $scope.resultsErrorMessage = "Sorry! We could not process your data. Please make sure it is correctly formatted and try again. You may wish to refer to the documentation via the sidebar for more information.";
            $scope.inProgress = false;
            
        });

    }

    $scope.downloadZipFile = function () {
        // Create a Blob from the base64-encoded zip content
        var zipBlob = new Blob([Uint8Array.from(atob($scope.zipContent), c => c.charCodeAt(0))], { type: 'application/zip' });

        // Create a download URL for the zip file
        var downloadUrl = (window.URL || window.webkitURL).createObjectURL(zipBlob);

        // Create a temporary anchor element
        var anchor = document.createElement('a');
        anchor.href = downloadUrl;
        anchor.download = $scope.zipFolderName;

        // Trigger the click event on the anchor element
        anchor.click();

        // Clean up the temporary anchor
        anchor.remove();
    };

    // HTTP request to run forager
    $scope.retrieveResults = function (file, selectedOption, selectedSwitch) {
        var payload = new FormData();
        payload.append('filename', file);
        
        if(selectedOption === 'get-sims' && $scope.OOVchoice !== 'process'){
            payload.append('selected-sims', 'sims');
        } 
        else if(selectedOption === 'get-sims' && $scope.OOVchoice === 'process'){
            console.log("sending process")
            payload.append('selected-process', 'process');
        } 
        else if(selectedOption === 'get-switch'){
            payload.append('selected-switch', selectedSwitch);
        }

        var payloadUrl = "/run-model" 

        var requestBody = {
            url: payloadUrl,
            method: 'POST',
            data: payload,
            headers: { 'Content-Type': undefined },
            transformRequest: angular.identity,
            responseType: 'arraybuffer'
        }

        // Generate zip file
        $http(requestBody).then(function (response) {
            var responseFile = new Blob([response.data], { type: 'application/zip' });
            $scope.downloadUrl = (window.URL || window.webkitURL).createObjectURL(responseFile);

            var anchor = document.createElement("a");
            anchor.download = $scope.zipFolderName;
            anchor.href = $scope.downloadUrl;
            $scope.resultsErrorMessage = "Success! Your download should begin shortly. Thanks for using forager!";
            anchor.click();
            $scope.inProgress = false;
            $scope.isLoadingResults = false;

            $scope.resultsErrorMessageStyle = {
                // set to dark green
                    color: '#006400',
                    fontWeight: 'bold'
                };

        }).catch(function (data) {

            $scope.resultsErrorMessageStyle = {
                    color: 'red',
                    fontWeight: 'bold'
                };
            $scope.resultsErrorMessage = "Sorry! We could not process the results due to a technical error. Please try again.";
            $scope.inProgress = false;
            $scope.isLoadingResults = false;
            
        });

        
    }
    
});
