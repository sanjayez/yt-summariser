const eventSource = new EventSource('http://localhost:8000/api/status-stream/3e253ef3-b5b4-43e0-93ed-ff32f222e0c9');

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Status update:', data);
    
    // Update your UI based on the data
    updateProgress(data);
    
    // Close connection when complete
    if (data.overall_status === 'success' || data.overall_status === 'failed') {
        eventSource.close();
    }
};

eventSource.onerror = function(error) {
    console.error('SSE error:', error);
    eventSource.close();
};