






require('dotenv').config(); 
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

mongoose
  .connect(process.env.MONGODB_URI, {
    useUnifiedTopology: true,
    useNewUrlParser: true,
  })
  .then(() => {
    console.log('MongoDB connected');
  })
  .catch((error) => console.log('Error connecting to MongoDB:', error));

const workerSchema = new mongoose.Schema({
  name: String,
  department: String,
  location: {
    type: { type: String, default: 'Point' },
    coordinates: [Number],
  },
  lastUpdated: { type: Date, default: Date.now },
});

workerSchema.index({ location: '2dsphere' }); // Geospatial index
const Worker = mongoose.model('Worker', workerSchema);

const siteSchema = new mongoose.Schema({
    siteName: { type: String, required: true },
    geofence: {
      center: { type: [Number], required: true },  // Latitude and Longitude
      radius: { type: Number, required: true }     // Radius in meters
    },
    assignedWorkers: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Worker' }]
  });
  

const Site = mongoose.model('Site', siteSchema);

// Create a new worker
app.post('/api/workers', async (req, res) => {
  const { name, department } = req.body;
  try {
    const newWorker = new Worker({
      name,
      department,
      location: {
        type: 'Point',
        coordinates: [0, 0] // Default location
      },
      lastUpdated: Date.now()
    });
    const savedWorker = await newWorker.save();
    res.json(savedWorker);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update worker location
app.post('/api/workers/:id/location', async (req, res) => {
  const { id } = req.params;
  const { latitude, longitude } = req.body;
  try {
    const worker = await Worker.findByIdAndUpdate(
      id,
      {
        location: {
          type: 'Point',
          coordinates: [longitude, latitude],
        },
        lastUpdated: Date.now(),
      },
      { new: true }
    );
    if (!worker) {
      return res.status(404).json({ error: 'Worker not found' });
    }
    return res.json(worker);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get all workers
app.get('/api/workers', async (req, res) => {
  try {
    const workers = await Worker.find();
    res.json(workers);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Create a construction site
// Create a construction site
app.post('/api/sites', async (req, res) => {
    const { siteName, geofence, assignedWorkers } = req.body;
  
    if (!geofence || !geofence.center || !geofence.radius) {
      return res.status(400).json({ error: 'Geofence must include both center and radius' });
    }
  
    try {
      const newSite = new Site({
        siteName,
        geofence: {
          center: geofence.center,
          radius: geofence.radius
        },
        assignedWorkers,
      });
  
      const savedSite = await newSite.save();
      res.json(savedSite);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

// Get all construction sites
app.get('/api/sites', async (req, res) => {
  try {
    const sites = await Site.find().populate('assignedWorkers');
    res.json(sites);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get workers within a geofence
app.get('/api/workers/nearby', async (req, res) => {
  const { latitude, longitude, radius } = req.query;
  if (!latitude || !longitude || !radius) {
    return res.status(400).json({ error: 'Please provide latitude, longitude, and radius' });
  }

  try {
    const workers = await Worker.find({
      location: {
        $geoWithin: {
          $centerSphere: [[longitude, latitude], radius / 6378100], // Radius in radians
        },
      },
    });
    res.json(workers);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});
app.patch('/api/sites/:id', async (req, res) => {
  const { id } = req.params;
  const { assignedWorkers } = req.body;
  
  try {
    const updatedSite = await Site.findByIdAndUpdate(
      id,
      { assignedWorkers },
      { new: true }
    ).populate('assignedWorkers');
    
    if (!updatedSite) {
      return res.status(404).json({ error: 'Site not found' });
    }
    
    res.json(updatedSite);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});